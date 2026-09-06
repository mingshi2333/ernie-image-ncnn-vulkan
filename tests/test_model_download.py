import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch
from tools.download_model import download
from tests.test_release_manifest import manifest

DATA=b'0123456789abcdef'*8192


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_GET(self):
                s=self.server;s.requests.append(dict(self.headers))
                offset=int(self.headers.get('Range','bytes=0-').split('=')[1].split('-')[0])
                mode=s.mode
                if mode=='error':
                    self.send_error(503);return
                etag='"v2"' if mode=='version' else '"v1"'
                if mode=='ignore':offset=0
                body=DATA[offset:]
                if mode=='wrong':body=body[:-1]+b'X'
                status=206 if self.headers.get('Range') and mode!='ignore' else 200
                self.send_response(status);self.send_header('ETag',etag)
                if status==206:self.send_header('Content-Range',f'bytes {offset+int(mode=="range")}-{len(DATA)-1}/{len(DATA)}')
                if mode!='truncated':self.send_header('Content-Length',str(len(body)))
                self.end_headers()
                if mode=='truncated':body=body[:4096]
                self.wfile.write(body)
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'target';self.root.mkdir()
        self.server.mode='normal';self.server.requests=[]
        self.m=manifest(f'http://127.0.0.1:{self.server.server_port}/'+ 'a'*40+'/data',DATA)
        self.dest=self.root/'weights/data.bin'
    def run_download(self):return download(self.m,self.root,emit=lambda _:None)
    def partial(self, data=DATA[:4096]):
        self.dest.parent.mkdir(exist_ok=True)
        Path(str(self.dest)+'.part').write_bytes(data)
        Path(str(self.dest)+'.part.json').write_text(json.dumps({'file':self.m['files'][0],'etag':'"v1"','last_modified':None}))
    def test_new_and_authenticated_skip(self):
        self.run_download();self.assertEqual(self.dest.read_bytes(),DATA)
        self.assertEqual(self.run_download()['files'][0]['status'],'verified_existing')
        self.assertEqual(len(self.server.requests),1)
    def test_resume_actual_range(self):
        self.partial();self.run_download();self.assertEqual(self.dest.read_bytes(),DATA)
        self.assertEqual(self.server.requests[0]['Range'],'bytes=4096-')
        self.assertEqual(self.server.requests[0]['If-Range'],'"v1"')
    def test_server_200_restarts_without_append(self):
        self.partial();self.server.mode='ignore';self.run_download();self.assertEqual(self.dest.read_bytes(),DATA)
    def test_version_change_preserves_partial(self):
        self.partial();self.server.mode='version'
        with self.assertRaisesRegex(ValueError,'version changed'):self.run_download()
        self.assertFalse(self.dest.exists());self.assertEqual(Path(str(self.dest)+'.part').read_bytes(),DATA[:4096])
    def test_truncated_then_resume(self):
        self.server.mode='truncated'
        with self.assertRaisesRegex(ValueError,'Truncated'):self.run_download()
        self.assertFalse(self.dest.exists());self.server.mode='normal';self.run_download()
        self.assertEqual(self.dest.read_bytes(),DATA)
    def test_wrong_last_byte_never_published(self):
        self.server.mode='wrong'
        with self.assertRaisesRegex(ValueError,'checksum'):self.run_download()
        self.assertFalse(self.dest.exists())
        with self.assertRaisesRegex(ValueError,'checksum'):self.run_download()
    def test_wrong_content_range(self):
        self.partial();self.server.mode='range'
        with self.assertRaisesRegex(ValueError,'Content-Range'):self.run_download()
        self.assertFalse(self.dest.exists())
    def test_existing_wrong_target_preserved(self):
        self.dest.parent.mkdir();self.dest.write_bytes(b'old')
        with self.assertRaisesRegex(ValueError,'Existing'):self.run_download()
        self.assertEqual(self.dest.read_bytes(),b'old');self.assertFalse(self.server.requests)
    def test_disk_capacity(self):
        with patch('tools.download_model.shutil.disk_usage',return_value=type('Disk',(),{'free':0})()):
            with self.assertRaisesRegex(OSError,'capacity'):self.run_download()
        self.assertFalse(self.server.requests)
    def test_symlink_and_resume_identity(self):
        self.dest.parent.mkdir();self.dest.symlink_to(self.root/'outside')
        with self.assertRaisesRegex(ValueError,'symlink'):self.run_download()
        self.dest.unlink();self.partial();self.m['files'][0]['sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'identity changed'):self.run_download()
    def test_atomic_publish_no_overwrite(self):
        self.partial(DATA)
        original=__import__('os').link
        def race(source,dest):
            dest.write_bytes(b'concurrent');return original(source,dest)
        with patch('tools.download_model.os.link',side_effect=race):
            with self.assertRaises(FileExistsError):self.run_download()
        self.assertEqual(self.dest.read_bytes(),b'concurrent')
    def test_http_failure_preserves_resume(self):
        self.partial();self.server.mode='error'
        with self.assertRaises(OSError):self.run_download()
        self.assertFalse(self.dest.exists());self.assertFalse((self.root/'.download.lock').exists())
        self.assertEqual(Path(str(self.dest)+'.part').read_bytes(),DATA[:4096])
    def test_hardlinked_resume_rejected(self):
        self.partial();other=self.root/'other'
        __import__('os').link(Path(str(self.dest)+'.part'),other)
        with self.assertRaisesRegex(ValueError,'hardlinked'):self.run_download()
        self.assertFalse(self.server.requests)
    def test_exclusive_download_lock(self):
        (self.root/'.download.lock').write_text('other process')
        with self.assertRaises(FileExistsError):self.run_download()
        self.assertEqual((self.root/'.download.lock').read_text(),'other process')
    def test_complete_partial_verified_without_network(self):
        self.partial(DATA);self.assertEqual(self.run_download()['files'][0]['status'],'verified_partial')
        self.assertFalse(self.server.requests)
