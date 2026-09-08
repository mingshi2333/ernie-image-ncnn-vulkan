"""Reviewed model compatibility; the runtime build still requires its exact pin."""


def compatible_model_revision(revision, lock):
    ncnn = lock['ncnn']
    return isinstance(revision, str) and (
        revision == ncnn['revision']
        or revision in ncnn.get('compatible_model_revisions', []))
