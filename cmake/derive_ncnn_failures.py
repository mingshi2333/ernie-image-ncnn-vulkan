#!/usr/bin/env python3
"""Compile-only fixes for the authenticated ncnn buffer execution path.

Keep upstream files intact. Unknown sources are rejected before any output is
written. Image allocators retain their original behavior; ERNIE executes VkMat
buffers. Buffer operations get cleanup and exact Vulkan OOM exceptions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


SHA = {
    "allocator": {
        "757342e625127b546ba715105dced695f15cd9f4f392d5875ef34178d0e2b86b": False,
        "af69e449e08fb6616ce629954edab900f68110b868a94c99a35fcef2ef791817": True,
    },
    "command": {"79283ad040e25983472ebabdaba85ab47cbc39e02adc568e62d6a22cb1443314": False},
    "net": {"258de463be0a3ad82072eea2fce714f4ebd3ed914f147732f05b9865e74ff57e": False},
}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def replace(text, before, after, count=1):
    actual = text.count(before)
    if actual != count:
        raise ValueError(f"Expected {count} occurrences of {before!r}, found {actual}")
    return text.replace(before, after)


def body_span(text, name):
    start = text.index("{", text.index(name))
    depth = 1
    pos = start + 1
    # Authenticated functions have balanced braces, including their comments.
    while depth:
        depth += (text[pos] == "{") - (text[pos] == "}")
        pos += 1
    return start, pos


def edit_function(text, name, edit):
    start, end = body_span(text, name)
    return text[:start] + edit(text[start:end]) + text[end:]


BUFFER_GUARD = r'''
#if NCNN_VULKAN
namespace {
// Only buffer allocations opt into throwing; image allocation APIs are unchanged.
thread_local unsigned ernie_ncnn_buffer_scope_depth = 0;
class ErnieNcnnBufferGuard
{
public:
    ErnieNcnnBufferGuard(const VkAllocator* owner, VkBufferMemory* pointer)
        : owner_(owner), pointer_(pointer) { ++ernie_ncnn_buffer_scope_depth; }
    ~ErnieNcnnBufferGuard()
    {
        if (pointer_)
        {
            if (pointer_->mapped_ptr) vkUnmapMemory(owner_->vkdev->vkdevice(), pointer_->memory);
            if (pointer_->buffer) vkDestroyBuffer(owner_->vkdev->vkdevice(), pointer_->buffer, 0);
            if (pointer_->memory)
            {
                vkFreeMemory(owner_->vkdev->vkdevice(), pointer_->memory, 0);
                ERNIE_METRIC_DESTROY
            }
            delete pointer_;
        }
        --ernie_ncnn_buffer_scope_depth;
    }
    void dismiss() { pointer_ = 0; }
private:
    const VkAllocator* owner_;
    VkBufferMemory* pointer_;
};
} // namespace
#endif
'''


def allocator(text, metrics=False):
    text = replace(text, '#include "allocator.h"', '#include "allocator.h"\n#include "ncnn_vulkan_failure.h"\n#include <memory>')
    hook = ("ernie::allocation_memory_destroyed(ernie_metric_device(owner_), "
            "ernie_metric_id(owner_), ernie_metric_id(pointer_->memory));") if metrics else ""
    helper = BUFFER_GUARD.replace("ERNIE_METRIC_DESTROY", hook)
    text = replace(text, "namespace ncnn {", "namespace ncnn {\n" + helper)

    for name in ["VkAllocator::create_buffer(", "VkAllocator::allocate_memory(",
                 "VkAllocator::allocate_dedicated_memory(", "VkAllocator::allocate_import_host_memory("]:
        def typed_failure(body, name=name):
            check = "throw_allocation_failure" if "import_host" in name else "check"
            return replace(body, "        return 0;", "        if (ernie_ncnn_buffer_scope_depth)\n"
                           f"            ernie_ncnn_detail::{check}(ret, \"Vulkan buffer allocation\");\n"
                           "        return 0;")
        text = edit_function(text, name, typed_failure)

    for cls in ["VkBlobAllocator", "VkWeightAllocator", "VkStagingAllocator", "VkWeightStagingAllocator"]:
        def guarded(body, cls=cls):
            block = cls in ("VkBlobAllocator", "VkWeightAllocator")
            owner = "block" if block else "ptr"
            body = replace(body, f"VkBufferMemory* {owner} = new VkBufferMemory;",
                           f"VkBufferMemory* {owner} = new VkBufferMemory();\n"
                           f"    ErnieNcnnBufferGuard buffer_guard(this, {owner});")
            # These original branches already release their Vulkan resources.
            body = body.replace(f"delete {owner};", f"buffer_guard.dismiss();\n        delete {owner};")
            if cls == "VkWeightAllocator":
                body = replace(body, 'NCNN_LOGE("vkCreateBuffer for weights failed %d", bufferResult);',
                               'NCNN_LOGE("vkCreateBuffer for weights failed %d", bufferResult);\n'
                               '        block->buffer = 0;\n'
                               '        ernie_ncnn_detail::check(bufferResult, "vkCreateBuffer for weights");')
                body = replace(body, 'NCNN_LOGE("vkGetMemoryHostPointerPropertiesEXT failed %d", ret);',
                               'NCNN_LOGE("vkGetMemoryHostPointerPropertiesEXT failed %d", ret);\n'
                               '                    ernie_ncnn_detail::check(ret, "vkGetMemoryHostPointerPropertiesEXT");')
                body = replace(body, "void* host_ptr = fastMalloc_with_alignment(memoryRequirements.size, d->buffer_offset_alignment);",
                               "void* host_ptr = fastMalloc_with_alignment(memoryRequirements.size, d->buffer_offset_alignment);\n"
                               "            std::unique_ptr<void, void (*)(void*)> host_guard(host_ptr, ncnn::fastFree);")
                body = replace(body, "ncnn::fastFree(host_ptr);", "host_guard.reset();", 2)
                body = replace(body, "d->host_ptrs.push_back(host_ptr);", "d->host_ptrs.push_back(host_ptr);\n                    host_guard.release();")
            body = replace(body, f"    {owner}->offset = 0;", f"    {owner}->offset = 0;\n"
                           f"    if (!{owner}->buffer) throw std::runtime_error(\"Vulkan buffer creation returned no buffer\");")
            body, binds = re.subn(r"(?m)^(\s*)vkBindBufferMemory\(([^;]+)\);$",
                                 r'\1ernie_ncnn_detail::check(vkBindBufferMemory(\2), "vkBindBufferMemory");', body)
            body, maps = re.subn(r"(?m)^(\s*)vkMapMemory\(([^;]+)\);$",
                                lambda m: m[1] + "VkResult map_result = vkMapMemory(" + m[2] + ");\n" + m[1] +
                                "if (map_result != VK_SUCCESS)\n" + m[1] + "{\n" + m[1] + "    " + owner +
                                "->mapped_ptr = 0;\n" + m[1] + '    ernie_ncnn_detail::check(map_result, "vkMapMemory");\n' + m[1] + "}", body)
            expected = 2 if cls == "VkWeightAllocator" else 1
            if binds != expected or maps != expected:
                raise ValueError(f"Unexpected {cls} bind/map count")
            if block:
                # The allocator owns a backing block after successful insertion.
                body = replace(body, "d->buffer_blocks.push_back(block);", "d->buffer_blocks.push_back(block);\n    buffer_guard.dismiss();")
                if cls == "VkWeightAllocator":
                    body = replace(body, "d->dedicated_buffer_blocks.push_back(block);", "d->dedicated_buffer_blocks.push_back(block);\n            buffer_guard.dismiss();")
                count = body.count("VkBufferMemory* ptr = new VkBufferMemory;")
                body = replace(body, "VkBufferMemory* ptr = new VkBufferMemory;",
                               "std::unique_ptr<VkBufferMemory> pointer_guard(new VkBufferMemory());\n"
                               "            VkBufferMemory* ptr = pointer_guard.get();", count)
                body = replace(body, "return ptr;", "pointer_guard.release();\n            return ptr;", count)
            else:
                body = replace(body, "return ptr;", "buffer_guard.dismiss();\n    return ptr;", 1 if cls == "VkWeightStagingAllocator" else 2)
                # The reused staging pointer precedes declaration of this guard.
                if cls == "VkStagingAllocator":
                    body = replace(body, "            buffer_guard.dismiss();\n    return ptr;", "            return ptr;", 1)
            return body
        text = edit_function(text, f"{cls}::fastMalloc(size_t size)", guarded)
    return text


def constructor(body, transfer=False):
    cleanup = '''        if (compute_command_fence) vkDestroyFence(vkdev->vkdevice(), compute_command_fence, 0);
        if (compute_command_pool) vkDestroyCommandPool(vkdev->vkdevice(), compute_command_pool, 0);'''
    if transfer:
        cleanup += '''
        if (upload_command_fence) vkDestroyFence(vkdev->vkdevice(), upload_command_fence, 0);
        if (upload_compute_semaphore) vkDestroySemaphore(vkdev->vkdevice(), upload_compute_semaphore, 0);
        if (transfer_command_pool) vkDestroyCommandPool(vkdev->vkdevice(), transfer_command_pool, 0);'''
    return replace(body, "    init();", '''    try
    {
        if (init() != 0) throw std::runtime_error("Vulkan command initialization failed");
    }
    catch (...)
    {
''' + cleanup + '''
        throw;
    }''')


def error_blocks(body, replacement):
    start = 0
    count = 0
    while True:
        pos = body.find("if (ret != VK_SUCCESS)", start)
        if pos == -1:
            return body, count
        left, right = body_span(body[pos:], "if (ret != VK_SUCCESS)")
        left += pos
        right += pos
        block = body[left:right]
        updated = replace(block, "return -1;", replacement)
        body = body[:left] + updated + body[right:]
        start = left + len(updated)
        count += 1


def command(text):
    text = replace(text, '#include "command.h"', '#include "command.h"\n#include "ncnn_vulkan_failure.h"')
    # A buffer failure can cancel recorded commands before submit_and_wait.
    # Free their owned payloads then, and clear pointers after normal encoding.
    payloads = [("copy_buffer", "regions"), ("copy_image", "regions"),
                ("copy_buffer_to_image", "regions"), ("copy_image_to_buffer", "regions"),
                ("push_constants", "values"), ("memory_barrers", "barriers"),
                ("buffer_barrers", "barriers"), ("image_barrers", "barriers")]
    cleanup = "static void ernie_ncnn_release_record(VkComputePrivate::record& record)\n{\n    switch (record.type)\n    {\n"
    for kind, field in payloads:
        cast = "(unsigned char*)" if kind == "push_constants" else ""
        cleanup += f"    case VkComputePrivate::record::TYPE_{kind}: delete[] {cast}record.{kind}.{field}; record.{kind}.{field} = 0; break;\n"
    cleanup += "    default: break;\n    }\n}\n\n"
    text = replace(text, "VkComputePrivate::VkComputePrivate(const VulkanDevice* _vkdev)",
                   cleanup + "VkComputePrivate::VkComputePrivate(const VulkanDevice* _vkdev)")
    text = edit_function(text, "VkComputePrivate::~VkComputePrivate()", lambda body: body[:1] +
                         "\n    for (size_t i = 0; i < delayed_records.size(); ++i) ernie_ncnn_release_record(delayed_records[i]);\n"
                         "    delayed_records.clear();\n" + body[1:])
    for cls in ["VkComputePrivate", "VkTransferPrivate"]:
        text = edit_function(text, f"{cls}::{cls}(const VulkanDevice*", lambda body: constructor(body, cls == "VkTransferPrivate"))
        for method in ["init", "begin_command_buffer", "end_command_buffer"]:
            def check_initialization(body, method=method):
                updated, count = error_blocks(body, 'ernie_ncnn_detail::throw_allocation_failure(ret, "Vulkan command initialization");\n            return -1;')
                if not count:
                    raise ValueError("Missing command initialization failure branches")
                if method == "init":
                    # Failed creation must not leave an undefined output handle
                    # for the constructor's partial-initialization cleanup.
                    pattern = r'(VkResult ret = vk(?:CreateCommandPool|AllocateCommandBuffers|CreateFence|CreateSemaphore)\([^;]+, &([a-z_]+)\);\s*if \(ret != VK_SUCCESS\)\s*\{)'
                    updated, handles = re.subn(pattern, lambda m: m[1] + "\n            " + m[2] + " = 0;", updated)
                    if handles != count:
                        raise ValueError("Unreviewed command handle initialization")
                    updated = replace(updated, "begin_command_buffer();", "if (begin_command_buffer() != 0) return -1;")
                return updated
            text = edit_function(text, f"{cls}::{method}()", check_initialization)

    text = edit_function(text, "VkTransfer::record_upload(", lambda body: _upload_guards(body))
    for cls in ["VkCompute", "VkTransfer"]:
        def submit(body):
            preamble = '''
    ernie_ncnn_detail::QueueLease compute_lease(vkdev, vkdev->info.compute_queue_family_index());
    ernie_ncnn_detail::QueueLease transfer_lease(vkdev, vkdev->info.transfer_queue_family_index());'''
            body = body[:1] + preamble + body[1:]
            body = replace(body, "vkdev->acquire_queue(vkdev->info.compute_queue_family_index())", "compute_lease.acquire()")
            if "vkdev->acquire_queue(vkdev->info.transfer_queue_family_index())" in body:
                body = replace(body, "vkdev->acquire_queue(vkdev->info.transfer_queue_family_index())", "transfer_lease.acquire()")
            body = re.sub(r"(?m)^\s*vkdev->reclaim_queue\([^;]+;\n", "\n", body)
            body, count = error_blocks(body, 'ernie_ncnn_detail::submission_failure(ret, compute_lease, transfer_lease);\n            return -1;')
            if count != (2 if cls == "VkCompute" else 5):
                raise ValueError("Unexpected submission error branches")
            body = body.replace("d->begin_command_buffer();", 'if (d->begin_command_buffer() != 0) throw std::runtime_error("Vulkan command begin failed");')
            body = replace(body, "d->end_command_buffer();", 'if (d->end_command_buffer() != 0) throw std::runtime_error("Vulkan command end failed");')
            if cls == "VkCompute":
                body = replace(body, "const VkComputePrivate::record& r = d->delayed_records[i];",
                               "VkComputePrivate::record& r = d->delayed_records[i];", 2)
                for kind, field in payloads:
                    old = "delete[](unsigned char*) r.push_constants.values;" if kind == "push_constants" else f"delete[] r.{kind}.{field};"
                    body = replace(body, old, "ernie_ncnn_release_record(r);")
            return body
        text = edit_function(text, f"{cls}::submit_and_wait()", submit)
    return text


def _upload_guards(body):
    body = body[:1] + '\n    if (src.empty()) throw std::runtime_error("Cannot upload an empty weight tensor");\n' + body[1:]
    body = replace(body, "    if (dst.empty())\n    {\n        return;\n    }",
                   '    if (dst.empty()) throw std::runtime_error("Weight allocation returned no buffer without a Vulkan OOM result");')
    body = replace(body, "    dst_staging.create_like(src_flattened, opt.staging_vkallocator);",
                   '    dst_staging.create_like(src_flattened, opt.staging_vkallocator);\n'
                   '    if (dst_staging.empty()) throw std::runtime_error("Weight staging allocation returned no buffer without a Vulkan OOM result");')
    body = replace(body, "    if (dst.allocator->mappable)\n    {",
                   '    if (dst.allocator->mappable)\n    {\n'
                   '        if (!dst.mapped_ptr()) throw std::runtime_error("Weight buffer has no host mapping");')
    body = replace(body, "    // memcpy src_flattened to staging",
                   '    if (!dst_staging.mapped_ptr()) throw std::runtime_error("Weight staging buffer has no host mapping");\n\n'
                   '    // memcpy src_flattened to staging')
    return body


def net(text):
    text = replace(text, '#include "net.h"', '#include "net.h"\n#include <memory>')
    def upload_owner(body):
        body = replace(body, "ncnn::VkTransfer* cmd_upload = 0;", "std::unique_ptr<ncnn::VkTransfer> cmd_upload;")
        body = replace(body, "cmd_upload = new ncnn::VkTransfer(d->vkdev);", "cmd_upload.reset(new ncnn::VkTransfer(d->vkdev));")
        return replace(body, "delete cmd_upload;", "cmd_upload.reset();")
    return edit_function(text, "Net::load_model(const DataReader& dr)", upload_owner)


def derive(source, allocator_path, output):
    source, allocator_path, output = Path(source).resolve(), Path(allocator_path).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must be outside the original ncnn source")
    inputs = {"allocator": allocator_path, "command": source / "src/command.cpp", "net": source / "src/net.cpp"}
    originals = {name: path.read_text() for name, path in inputs.items()}
    identities = {name: digest(text) for name, text in originals.items()}
    for name, value in identities.items():
        if value not in SHA[name]:
            raise ValueError(f"Unreviewed ncnn {name} source: {value}")
    transformed = {
        "allocator": allocator(originals["allocator"], SHA["allocator"][identities["allocator"]]),
        "command": command(originals["command"]),
        "net": net(originals["net"]),
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, contents in transformed.items():
        path = output / f"{name}.cpp"
        if not path.exists() or path.read_text() != contents:
            path.write_text(contents)
    header = Path(__file__).with_name("ncnn_vulkan_failure.h")
    contents = header.read_text()
    target = output / header.name
    if not target.exists() or target.read_text() != contents:
        target.write_text(contents)
    provenance = {"source_sha256": identities, "derived_sha256": {name: digest(text) for name, text in transformed.items()},
                  "patcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "header_sha256": digest(contents),
                  "scope": "VkMat buffer allocation, transfer/compute failure propagation, exception-safe Net upload ownership; image allocators unchanged"}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("allocator", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(derive(args.source, args.allocator, args.output)))
