import os
import sys
import time
import warnings
import hashlib
import math
import onnxruntime as ort
import numpy as np
from pathlib import Path
import signal
import tempfile
import shutil
import subprocess
import random
import json
import mmap, gc
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
# from tisw import byte_entropy_histogram, fml_bit_features, byte_frequency
# 这里直接将tisw完整复制过来
import ctypes
from ctypes import wintypes

# 加载 DLL
_dll_path = os.path.join(os.path.dirname(__file__), "TiSDLL.dll")
_lib = ctypes.CDLL(_dll_path)

# ---------- 类型别名 ----------
_u8p   = ctypes.POINTER(ctypes.c_uint8)
_f32p  = ctypes.POINTER(ctypes.c_float)
_f64p  = ctypes.POINTER(ctypes.c_double)
_i32p  = ctypes.POINTER(ctypes.c_int32)
_i64p  = ctypes.POINTER(ctypes.c_int64)
_int   = ctypes.c_int

# ---------- 函数签名 ----------
_lib.byte_entropy_histogram.argtypes = [_u8p, _int, _f32p]
_lib.byte_entropy_histogram.restype  = None

_lib.tex_patch_stats.argtypes = [_u8p, _int, _int, _f32p]
_lib.tex_patch_stats.restype  = None

_lib.fml_bit_features.argtypes = [_u8p, _int, _i32p, _i64p, _i64p, _i64p]
_lib.fml_bit_features.restype  = None

_lib.byte_frequency.argtypes = [_u8p, _int, _f32p]
_lib.byte_frequency.restype  = None

# ---------- 辅助：拿 numpy 数组的 ctypes 指针 ----------
def _as(arr, typ):
    return arr.ctypes.data_as(typ)

# ============================================================
#  1. byte_entropy_histogram
#     输入: bytes 或 bytearray
#     输出: np.float32 shape (512,)   归一化 2D 直方图
# ============================================================
def byte_entropy_histogram(data: bytes) -> np.ndarray:
    buf = np.frombuffer(data, dtype=np.uint8)
    out = np.empty(512, dtype=np.float32)
    _lib.byte_entropy_histogram(
        _as(buf, _u8p),
        ctypes.c_int(len(buf)),
        _as(out, _f32p),
    )
    return out

# ============================================================
#  2. tex_patch_stats
#     输入: raw bytes (uint8), n_patches, patch_size
#     输出: mean, std, mx, mn  各 shape (n_patches, 3)
# ============================================================
def tex_patch_stats(raw_bytes: bytes, n_patches: int, patch_size: int):
    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    out = np.empty(n_patches * 12, dtype=np.float32)
    _lib.tex_patch_stats(
        _as(buf, _u8p),
        ctypes.c_int(n_patches),
        ctypes.c_int(patch_size),
        _as(out, _f32p),
    )
    # 重排: DLL 输出的是 [mean_c0, mean_c1, mean_c2, std_c0..., max_c0..., min_c0...] per patch
    out = out.reshape(n_patches, 12)
    m   = out[:, 0:3]
    std = out[:, 3:6]
    mx  = out[:, 6:9]
    mn  = out[:, 9:12]
    return m, std, mx, mn

# ============================================================
#  3. fml_bit_features
#     输入: raw bytes (uint8), 长度必须是 10 的倍数
#     输出: count(int32), post_total(int64),
#           post_first(int64), post_last(int64)
#           各 shape (n_rows,)
# ============================================================
def fml_bit_features(raw_bytes: bytes, n_rows: int):
    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    count = np.empty(n_rows, dtype=np.int32)
    pt    = np.empty(n_rows, dtype=np.int64)
    pf    = np.empty(n_rows, dtype=np.int64)
    pl    = np.empty(n_rows, dtype=np.int64)
    _lib.fml_bit_features(
        _as(buf, _u8p),
        ctypes.c_int(n_rows),
        _as(count, _i32p),
        _as(pt, _i64p),
        _as(pf, _i64p),
        _as(pl, _i64p),
    )
    return count, pt, pf, pl

# ============================================================
#  4. byte_frequency
#     输入: bytes
#     输出: np.float32 shape (256,) 归一化字节频率
# ============================================================
def byte_frequency(data: bytes) -> np.ndarray:
    buf = np.frombuffer(data, dtype=np.uint8)
    out = np.empty(256, dtype=np.float32)
    _lib.byte_frequency(
        _as(buf, _u8p),
        ctypes.c_int(len(buf)),
        _as(out, _f32p),
    )
    return out


import lief

lief.logging.disable()

# ===================== 常量 / 开关 =====================
TARGET_INSTR = 59000
STRIDE = 3
REQUIRED_BYTES = TARGET_INSTR * STRIDE  # 177000

if os.path.exists("tis_dbg"):
    Debug = 1
else:
    Debug = 0

if os.path.exists("tis_swap"):
    Swap = 1
    print("反转模式已开启，文件将被收集")
else:
    Swap = 0
zxProbFix = 0.995
FMLProbFix = 1.0
TEXProbFix = 1.01

CPU_BATCH_SIZE = 3
DML_BATCH_SIZE = 1

BATCH_WAIT_MIN = 0.002
BATCH_WAIT_MAX = 0.180
BATCH_WAIT_FACTOR = 0.35

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONV_MODEL = os.path.join(BASE_DIR, 'tisave.onnx')
ZX_MODEL   = os.path.join(BASE_DIR, 'TisaveAst.onnx')
FML_MODEL  = os.path.join(BASE_DIR, 'TisaveFML.onnx')
TEX_MODEL  = os.path.join(BASE_DIR, 'TisaveTEX.onnx')
VOCAB_PATH = os.path.join(BASE_DIR, 'vocab.json')


TRIGRAM_MAX  = 16777216  # 256^3
SEQ_LEN      = 59000

warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names.*")
ort.set_default_logger_severity(4)

print_lock = threading.Lock()

# ===================== PBC LUT（全局，启动时加载一次） =====================
_PBC_LUT = None

def load_pbc_lut():
    global _PBC_LUT
    if _PBC_LUT is not None:
        return _PBC_LUT
    if not os.path.exists(VOCAB_PATH):
        print(f"[错误] vocab.json 未找到: {VOCAB_PATH}")
        sys.exit(1)
    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f)
    lut = np.zeros(TRIGRAM_MAX, dtype=np.int16)
    for k, v in vocab.items():
        lut[int(k)] = v
    _PBC_LUT = lut
    return _PBC_LUT

# ===================== TEX 特征提取（PBC版） =====================
def extract_tex_from_data(data, pe):
    try:
        if pe is None:
            return None

        if len(pe.sections) == 0:
            return None

        sections = []
        for section in pe.sections:
            raw_name = section.name.strip('\x00').lower().lstrip('.')
            content_len = len(section.content)
            has_code_flag = bool(section.characteristics & 0x20)
            sections.append((raw_name, content_len, has_code_flag, section))

        target_section = None

        CODE_SECTION_NAMES = {
            'text', 'code', 'icode', 'itext',
            'init', 'page', 'pagelk',
            'textbss', 'text2', 'stub', 'fp_text', 'orpc',
            'upx0', 'upx1', 'upx2',
            'vmp0', 'vmp1', 'vmp2',
            'themida', 'aspack', 'adata', 'nsp0', 'nsp1',
            'petite', 'pecompact', 'morphine', 'securom',
            'maskpe', 'jdpack',
        }
        for name, clen, _, sec in sections:
            if name in CODE_SECTION_NAMES and clen > 0:
                target_section = sec
                break

        if target_section is None:
            ep = pe.optional_header.addressof_entrypoint
            if ep > 0:
                for name, clen, _, sec in sections:
                    va = sec.virtual_address
                    vs = sec.virtual_size
                    if va <= ep < va + vs and clen > 0:
                        target_section = sec
                        break

        if target_section is None:
            best_size = -1
            for name, clen, has_code, sec in sections:
                if has_code and clen > best_size:
                    best_size = clen
                    target_section = sec

        if target_section is None:
            best_size = -1
            for name, clen, _, sec in sections:
                if clen > best_size:
                    best_size = clen
                    target_section = sec

        if target_section is None:
            return None

        raw_data = bytes(target_section.content)
        if len(raw_data) == 0:
            return None

        buf = np.zeros(REQUIRED_BYTES, dtype=np.uint8)
        copy_len = min(len(raw_data), REQUIRED_BYTES)
        buf[:copy_len] = np.frombuffer(raw_data[:copy_len], dtype=np.uint8)

        trigrams = buf.reshape(SEQ_LEN, 3)

        keys = (trigrams[:, 0].astype(np.int32) * 65536 +
                trigrams[:, 1].astype(np.int32) * 256   +
                trigrams[:, 2].astype(np.int32))

        lut = load_pbc_lut()
        tokens = lut[keys].astype(np.float32)
        return tokens

    except Exception:
        return None

# ===================== 工具函数 =====================
def dbg(msg):
    if Debug:
        with print_lock:
            print(msg, flush=True)

dbg("DEBUG模式已启用，你将收到更多的调试信息")

time.sleep(0.2)
print('-' * 40)
print('Ti Security 1.29')
print('Packed time: 2026/07/23 10:48')
print('引擎信息:')
print('| Ti 卷积核 20260723')
print('| Ti 智芯 20260723')
print('| Ti FML 20260723')
print('| Ti TEX 20260723')
print('-' * 20)
print('Feedback Email: 1828529634@qq.com')
print('-' * 40)
print('开-始-扫-描')
time.sleep(0.3)

def parse_pe_once(data):
    if len(data) < 64 or data[:2] != b'MZ':
        return None
    try:
        cfg = lief.PE.ParserConfig()
        cfg.parse_signature = False
        cfg.parse_exports   = False
        cfg.parse_reloc     = False
        cfg.parse_rsrc      = False
        cfg.parse_imports   = True
        return lief.PE.parse(data, cfg)
    except:
        return None

def clear_line():
    sys.stdout.write('\033[2K\r')
    sys.stdout.flush()

def print_status(scanned, threats_count, current_file):
    with print_lock:
        clear_line()
        if scanned % 1000 == 0:
            gc.collect()
        status = f"扫描状态: 已扫描 {scanned} 文件 | 威胁: {threats_count} | 当前: {current_file[:20]}{'...' if len(current_file) > 20 else ''}"
        sys.stdout.write(status)
        sys.stdout.flush()

def format_time(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}"

def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out.astype(np.float32)

# ===================== TaskQueue =====================
class TaskQueue:
    def __init__(self, name=""):
        self.name = name
        self._dq = deque()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._finished = False

    def put(self, item):
        with self._cond:
            self._dq.append(item)
            self._cond.notify()

    def put_front(self, item):
        with self._cond:
            self._dq.appendleft(item)
            self._cond.notify()

    def put_many(self, items):
        if not items:
            return
        with self._cond:
            self._dq.extend(items)
            self._cond.notify_all()

    def get(self, timeout=None):
        with self._cond:
            if timeout is None:
                while not self._dq:
                    if self._finished:
                        return None
                    self._cond.wait(timeout=0.5)
                return self._dq.popleft()
            end = time.time() + max(0.0, float(timeout))
            while not self._dq:
                if self._finished:
                    return None
                remaining = end - time.time()
                if remaining <= 0:
                    return "TIMEOUT"
                self._cond.wait(timeout=min(0.5, remaining))
            return self._dq.popleft()

    def size(self):
        with self._lock:
            return len(self._dq)

    def pop_newest(self, n=2):
        with self._lock:
            out = []
            for _ in range(min(n, len(self._dq))):
                out.append(self._dq.pop())
            return out

    def finish(self):
        with self._cond:
            self._finished = True
            self._cond.notify_all()

def rebalance(q_cpu: TaskQueue, q_dml: TaskQueue):
    s_cpu = q_cpu.size()
    s_dml = q_dml.size()
    if s_cpu > s_dml * 1.5 and s_cpu > 10:
        moved = q_cpu.pop_newest(6)
        q_dml.put_many(moved)
        dbg(f"负载均衡: CPU({s_cpu}) -> DML({s_dml}) moved={len(moved)}")
    elif s_dml > s_cpu * 1.5 and s_dml > 10:
        moved = q_dml.pop_newest(6)
        q_cpu.put_many(moved)
        dbg(f"负载均衡: DML({s_dml}) -> CPU({s_cpu}) moved={len(moved)}")

# ===================== 压缩包 =====================
def isArchive(path):

    signatures = {
        b'\x50\x4B\x03\x04', b'\x37\x7A\xBC\xAF', b'\x52\x61\x72\x21',
        b'\x1F\x8B\x08',     b'\x42\x5A\x68',     b'\xFD\x37\x7A\x58',
        b'\x4D\x53\x43\x46', b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1',
    }
    maxlen = max(len(s) for s in signatures)
    try:
        with open(path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            header = mm.read(maxlen)
            mm.close()
            return int(any(header.startswith(s) for s in signatures))
    except:
        return 0

def RecursiveArchive(archivePath, unpackPath):
    if not os.path.exists(unpackPath):
        os.makedirs(unpackPath, exist_ok=True)
    MAX_DEPTH = 20
    temp_root = os.path.join(unpackPath, "__temp")
    os.makedirs(temp_root, exist_ok=True)

    def safe_rm_rf(path):
        try:
            shutil.rmtree(path, ignore_errors=True)
        except:
            pass

    def _extract(path, depth):
        if depth > MAX_DEPTH:
            return
        tmpdir = os.path.join(temp_root, f"level_{depth}_{os.path.basename(path)}")
        os.makedirs(tmpdir, exist_ok=True)
        try:
            result = subprocess.run(
                ["7z", "x", "-y", f"-o{tmpdir}", path,"-mmt=on"],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60
            )
            if result.returncode != 0:
                safe_rm_rf(tmpdir)
                return
        except Exception:
            safe_rm_rf(tmpdir)
            return
        try:
            for root, _, files in os.walk(tmpdir):
                for f in files:
                    fp = os.path.realpath(os.path.join(root, f))
                    if isArchive(fp) != 0:
                        _extract(fp, depth + 1)
        except Exception:
            pass
        try:
            for root, _, files in os.walk(tmpdir):
                for f in files:
                    src = os.path.join(root, f)
                    rel = os.path.relpath(src, tmpdir)
                    dst = os.path.join(unpackPath, rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.move(src, dst)
        except Exception:
            pass
        safe_rm_rf(tmpdir)

    _extract(archivePath, 0)
    safe_rm_rf(temp_root)

# ===================== 特征提取 =====================
def extract_fml_from_data(data):
    try:
        if data[:2] != b'MZ':
            return None
        file_size = len(data)
        if file_size < 100:
            return None
        block_size = max(1, file_size // 100)
        half_block = math.floor(block_size / 2)
        all_batches = bytearray()
        for i in range(100):
            start_pos = i * block_size
            chunk_start = data[start_pos:start_pos + 250].ljust(250, b'\x00')
            hp = start_pos + half_block
            chunk_half = data[hp:hp + 250].ljust(250, b'\x00')
            all_batches.extend(chunk_start)
            all_batches.extend(chunk_half)
            all_batches.extend(chunk_start[:10])
        raw = bytes(all_batches)
        n_rows = 5100
        t_count, t_post_total, t_post_first, t_post_last = fml_bit_features(raw, n_rows)
        features = np.column_stack((
            t_count.astype(np.float32),
            t_post_total.astype(np.float32),
            t_post_first.astype(np.float32),
            t_post_last.astype(np.float32),
        ))
        return features.T.astype(np.float32)[np.newaxis, :, :]
    except Exception:
        return None

class FeatureExtractor:
    def __init__(self):
        self.api_list = []
        api_path = os.path.join(BASE_DIR, 'win32api.txt')
        if os.path.exists(api_path):
            with open(api_path, 'r', encoding='utf-8', errors='ignore') as f:
                self.api_list = sorted([line.strip() for line in f if line.strip()])
        self.api_map = {api: i for i, api in enumerate(self.api_list)}

    def extract_from_data(self, data, pe):
        try:
            if pe is None:
                return None
            be_feat   = byte_entropy_histogram(data)
            byte_freq = byte_frequency(data)
            iat_feat  = np.zeros(256, dtype=np.float32)
            api_feat  = np.zeros(len(self.api_list), dtype=np.float32)
            _amap = self.api_map
            _md5  = hashlib.md5
            for lib in pe.imports:
                try:
                    dname = lib.name.lower()
                    dh = _md5(dname.encode()).digest()[-1]
                except:
                    dh = 0
                for entry in lib.entries:
                    if entry.name:
                        try:
                            fname = entry.name
                            fh = _md5(fname.encode()).digest()[-1]
                            iat_feat[(dh + fh) & 0xFF] = 1.0
                            ai = _amap.get(fname)
                            if ai is not None:
                                api_feat[ai] = 1.0
                        except:
                            continue
            meta_feat = np.zeros(256, dtype=np.float32)
            oh = pe.optional_header
            hd = pe.header
            for i, v in enumerate((
                hd.time_date_stamps, oh.addressof_entrypoint,
                oh.imagebase % 1000000, oh.section_alignment % 10000,
                oh.file_alignment % 10000, hd.numberof_sections,
                oh.sizeof_image % 1000000, oh.dll_characteristics,
            )):
                meta_feat[(i * 31) % 256] = (int(v) % 10000) / 10000.0
            return np.concatenate([be_feat, iat_feat, meta_feat, byte_freq, api_feat]).astype(np.float32)
        except Exception:
            return None

ZX_MAX_BYTES = 30000
def extract_zx_from_data(data):
    
    try:
        
        if len(data) < 64 or data[:2] != b'MZ':return None
        data = data[:ZX_MAX_BYTES] 
        if len(data) < ZX_MAX_BYTES:
            data = data.ljust(ZX_MAX_BYTES, b'\x00')
        
        # 使用 uint32 防止溢出，计算 3-gram

        arr = np.frombuffer(data, dtype=np.uint8).astype(np.uint32).reshape(10000, 3)

        features = arr[:, 0] * np.uint32(65536) + arr[:, 1] * np.uint32(256) + arr[:, 2]

        return features.astype(np.float32)
    except: return None

# ===================== 融合 =====================
def SigmoidValues(a, b):
    w_a = abs(a - 0.5)
    w_b = abs(b - 0.5)
    denom = w_a + w_b
    if denom == 0:
        return 0.5
    return abs((w_a * a + w_b * b) / denom)

def FixValues(a, b):
    return a + (a - 0.5) * (b - 1)

# ===================== 签名白名单 =====================
class _WinTrust:
    """Encapsulates all WinVerifyTrust ctypes plumbing."""

    # --- WinTrust constants ---
    WTD_UI_NONE = 2
    WTD_REVOKE_NONE = 0
    WTD_CHOICE_FILE = 1
    WTD_STATEACTION_VERIFY = 1
    WTD_STATEACTION_CLOSE = 2
    WTD_SAFER_FLAG = 0x100
    ERROR_SUCCESS = 0

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", ctypes.c_ulong),
            ("Data2", ctypes.c_ushort),
            ("Data3", ctypes.c_ushort),
            ("Data4", ctypes.c_ubyte * 8),
        ]

        def __init__(self, d1, d2, d3, d4):
            self.Data1 = d1
            self.Data2 = d2
            self.Data3 = d3
            for i in range(8):
                self.Data4[i] = d4[i]

    class WINTRUST_FILE_INFO(ctypes.Structure):
        _fields_ = [
            ("cbStruct", wintypes.DWORD),
            ("pcwszFilePath", wintypes.LPCWSTR),
            ("hFile", wintypes.HANDLE),
            ("pgKnownSubject", ctypes.c_void_p),
        ]

    class WINTRUST_DATA(ctypes.Structure):
        _fields_ = [
            ("cbStruct", wintypes.DWORD),
            ("pPolicyCallbackData", ctypes.c_void_p),
            ("pSIPClientData", ctypes.c_void_p),
            ("dwUIChoice", wintypes.DWORD),
            ("fdwRevocationChecks", wintypes.DWORD),
            ("dwUnionChoice", wintypes.DWORD),
            ("pFile", ctypes.c_void_p),
            ("dwStateAction", wintypes.DWORD),
            ("hWVTStateData", wintypes.HANDLE),
            ("pwszURLReference", wintypes.LPCWSTR),
            ("dwProvFlags", wintypes.DWORD),
            ("dwUIContext", wintypes.DWORD),
            ("pSignatureSettings", ctypes.c_void_p),
        ]

    def __init__(self):
        # WINTRUST_ACTION_GENERIC_VERIFY_V2
        # {00AAC56B-CD44-11d0-8CC2-00C04FC295EE}
        self._action = self.GUID(
            0x00AAC56B, 0xCD44, 0x11D0,
            (0x8C, 0xC2, 0x00, 0xC0, 0x4F, 0xC2, 0x95, 0xEE),
        )
        self._fn = ctypes.WinDLL("wintrust.dll").WinVerifyTrust
        self._fn.restype = wintypes.LONG
        self._fn.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p]

    def _build_data(self, path, state_action):
        file_info = self.WINTRUST_FILE_INFO()
        file_info.cbStruct = ctypes.sizeof(self.WINTRUST_FILE_INFO)
        file_info.pcwszFilePath = path
        file_info.hFile = None
        file_info.pgKnownSubject = None

        wtd = self.WINTRUST_DATA()
        wtd.cbStruct = ctypes.sizeof(self.WINTRUST_DATA)
        wtd.dwUIChoice = self.WTD_UI_NONE
        wtd.fdwRevocationChecks = self.WTD_REVOKE_NONE
        wtd.dwUnionChoice = self.WTD_CHOICE_FILE
        wtd.pFile = ctypes.cast(ctypes.pointer(file_info), ctypes.c_void_p)
        wtd.dwStateAction = state_action
        wtd.dwProvFlags = self.WTD_SAFER_FLAG
        # keep file_info alive alongside wtd
        wtd._file_info = file_info
        return wtd

    def _call(self, wtd):
        return self._fn(None, ctypes.byref(self._action), ctypes.byref(wtd))

    def verify(self, path):
        """Return True if the file at `path` has a trusted signature."""
        try:
            wtd = self._build_data(path, self.WTD_STATEACTION_VERIFY)
            status = self._call(wtd)

            # Always release the state data.
            wtd.dwStateAction = self.WTD_STATEACTION_CLOSE
            self._call(wtd)

            return status == self.ERROR_SUCCESS
        except Exception:
            return False


_wintrust = _WinTrust()


def WinVerifyTrust(path):
    """Verify the Authenticode signature of a PE file via WinVerifyTrust.

    Args:
        path: path to the PE file (assumed to be a valid PE).

    Returns:
        True  -> signature present and trusted by system policy.
        False -> signature missing, invalid, or untrusted.
    """
    return _wintrust.verify(path)


_SIG_WHITELIST = set()
_HASH_WHITELIST = set()
_SIG_WL_LOCK = threading.Lock()
_SIG_CACHE = {}
_SIG_CACHE_MAX = 4096
_SIG_BUILT = False

_CORE_SEED_FILES = [
    "explorer.exe",
    r"System32\smss.exe", r"System32\csrss.exe", r"System32\wininit.exe",
    r"System32\winlogon.exe", r"System32\services.exe", r"System32\lsass.exe",
    r"System32\svchost.exe", r"System32\taskhostw.exe", r"System32\dllhost.exe",
    r"System32\RuntimeBroker.exe", r"System32\conhost.exe", r"System32\sihost.exe",
    r"System32\ctfmon.exe", r"System32\spoolsv.exe", r"System32\ntoskrnl.exe",
    r"System32\ntdll.dll", r"System32\kernel32.dll", r"System32\kernelbase.dll",
    r"System32\user32.dll", r"System32\gdi32.dll", r"System32\shell32.dll",
    r"System32\combase.dll",
]

EXTERNAL_WHITE_LIST = [
    'BUSINESS_CATEGORY=Private Organization, serialNumber=911101026662879416, JURISDICTION_OF_INCORPORATION_C=CN, JURISDICTION_OF_INCORPORATION_SP=Beijing, C=CN, ST=Beijing, L=Beijing, STREET_ADDRESS=朝阳区酒仙桥路6号院2号楼1至19层104号内8层801, O=Beijing Qihu Technology Co.\\, Ltd., CN=Beijing Qihu Technology Co.\\, Ltd.',
    'BUSINESS_CATEGORY=Private Organization, serialNumber=91110105MA018WY219, JURISDICTION_OF_INCORPORATION_C=CN, JURISDICTION_OF_INCORPORATION_SP=BEIJING, C=CN, ST=Beijing, L=Beijing, STREET_ADDRESS=朝阳区创远路36号院15号楼1层101室, O=Vance Technology, CN=Vance Technology',
    'BUSINESS_CATEGORY=Private Organization, serialNumber=91430111MAEXLE0M5P, JURISDICTION_OF_INCORPORATION_C=CN, JURISDICTION_OF_INCORPORATION_SP=Hunan, JURISDICTION_OF_INCORPORATION_L=Changsha, C=CN, ST=Hunan, L=Changsha, O=长沙伏特旷野科技有限公司, CN=长沙伏特旷野科技有限公司',
    'C=AM, ST=Armavir, L=Armavir, O=FSPRO LLC, CN=FSPRO LLC',
    'C=CN, O=TiSecurity, CN=Hashcake',
    'C=CN, ST=Beijing, O=Antiy Labs, CN=Antiy Labs',
    'C=CN, ST=Beijing, O=Beijing Duyou Science and Technology Co.\\,Ltd., CN=Beijing Duyou Science and Technology Co.\\,Ltd.',
    'C=CN, ST=Guangdong Province, L=Shenzhen, O=Sangfor Technologies Inc., CN=Sangfor Technologies Inc.',
    'C=CN, ST=Guangdong, L=Jiangmen, O=Jiangmen Maidi E-commerce Co.\\, Ltd., CN=Jiangmen Maidi E-commerce Co.\\, Ltd.',
    'C=CN, ST=Shanghai, O=miHoYo Co.\\,Ltd., CN=miHoYo Co.\\,Ltd.',
    'C=CN, ST=北京市, O=Beijing iQIYI Science & Technology Co.\\, Ltd., CN=Beijing iQIYI Science & Technology Co.\\, Ltd.',
    'C=CN, ST=广东省, L=珠海市, O=Zhuhai Kingsoft Office Software Co.\\, Ltd., CN=Zhuhai Kingsoft Office Software Co.\\, Ltd.',
    'C=CZ, L=Praha, O=Avast Software s.r.o., OU=RE 999, CN=Avast Software s.r.o.',
    'C=HK, ST=Kowloon, L=Kwun Tong, O=AOMEI International Network Limited, CN=AOMEI International Network Limited',
    'C=RO, L=Bucuresti, O=Bitdefender SRL, OU=DEVSUP EPSINSTALLER, CN=Bitdefender SRL',
    'C=US, ST=Arizona, O=Gen Digital Inc., CN=Gen Digital Inc.',
    'C=US, ST=California, L=Redwood City, O=Oracle America\\, Inc., OU=Software Engineering, CN=Oracle America\\, Inc.',
    'C=US, ST=California, L=San Francisco, O=Anysphere\\, Inc., CN=Anysphere\\, Inc.',
    'C=US, ST=California, L=San Francisco, O=Mozilla Corporation, OU=Firefox Engineering Operations, CN=Mozilla Corporation',
    'C=US, ST=California, L=San Francisco, O=Notion Labs\\, Inc., CN=Notion Labs\\, Inc.',
    'C=US, ST=California, L=San Francisco, O=OpenJS Foundation, CN=OpenJS Foundation',
    'C=US, ST=California, L=Santa Clara, O=NVIDIA Corporation, OU=1-F, CN=NVIDIA Corporation',
    'C=US, ST=Colorado, L=Denver, O=Zed Industries Inc, CN=Zed Industries Inc',
    'C=US, ST=New Hampshire, L=Wolfeboro, O=Python Software Foundation, CN=Python Software Foundation',
    'C=US, ST=Oregon, L=Beaverton, O=Python Software Foundation, CN=Python Software Foundation',
    'C=US, ST=Washington, L=Bellevue, O=Valve Corp., CN=Valve Corp.',
    'C=US, ST=Washington, L=Redmond, O=Microsoft Corporation, CN=Microsoft Corporation',
    'JURISDICTION_OF_INCORPORATION_C=CA, BUSINESS_CATEGORY=Private Organization, serialNumber=1131559-5, C=CA, ST=Ontario, L=Toronto, O=Tailscale Inc., CN=Tailscale Inc.',
    'JURISDICTION_OF_INCORPORATION_C=CN, JURISDICTION_OF_INCORPORATION_SP=Guangdong Province, JURISDICTION_OF_INCORPORATION_L=Guangzhou, BUSINESS_CATEGORY=Private Organization, serialNumber=914401016756828477, C=CN, ST=Guangdong Province, L=Guangzhou, O=Guangzhou Shirui Electronics Co.\\, Ltd., CN=Guangzhou Shirui Electronics Co.\\, Ltd.',
    'JURISDICTION_OF_INCORPORATION_C=CN, JURISDICTION_OF_INCORPORATION_SP=Guangdong Province, JURISDICTION_OF_INCORPORATION_L=Shenzhen, BUSINESS_CATEGORY=Private Organization, serialNumber=9144030071526726XG, C=CN, ST=Guangdong Province, L=Shenzhen, O=Tencent Technology (Shenzhen) Company Limited, CN=Tencent Technology (Shenzhen) Company Limited',
    'JURISDICTION_OF_INCORPORATION_C=CN, JURISDICTION_OF_INCORPORATION_SP=Guangdong Province, JURISDICTION_OF_INCORPORATION_L=Shenzhen, BUSINESS_CATEGORY=Private Organization, serialNumber=91440300746612636Q, C=CN, ST=Guangdong Province, L=Shenzhen, O=Shenzhen Xunlei Network Technology Co.\\, Ltd., CN=Shenzhen Xunlei Network Technology Co.\\, Ltd.',
    'JURISDICTION_OF_INCORPORATION_C=CN, JURISDICTION_OF_INCORPORATION_SP=Zhejiang, JURISDICTION_OF_INCORPORATION_L=Yuhang District, BUSINESS_CATEGORY=Private Organization, serialNumber=91330110MA2B00R29G, C=CN, ST=Zhejiang, L=Hangzhou, O=DingTalk Technology Co.\\,Ltd., CN=DingTalk Technology Co.\\,Ltd.',
    'JURISDICTION_OF_INCORPORATION_C=CN, JURISDICTION_OF_INCORPORATION_SP=上海市, BUSINESS_CATEGORY=Private Organization, serialNumber=91310110787862412B, C=CN, ST=上海市, O=上海贝锐信息科技股份有限公司, CN=上海贝锐信息科技股份有限公司',
    'JURISDICTION_OF_INCORPORATION_C=CN, JURISDICTION_OF_INCORPORATION_SP=上海市, JURISDICTION_OF_INCORPORATION_L=奉贤区, BUSINESS_CATEGORY=Private Organization, serialNumber=9131012030160721XG, C=CN, ST=上海市, O=Shanghai Microvirt Software Technology Co.\\, Ltd., CN=Shanghai Microvirt Software Technology Co.\\, Ltd.',
    'JURISDICTION_OF_INCORPORATION_C=CN, JURISDICTION_OF_INCORPORATION_SP=北京市, JURISDICTION_OF_INCORPORATION_L=石景山区, BUSINESS_CATEGORY=Private Organization, serialNumber=91110107599635562F, C=CN, ST=北京市, O=Douyin Vision Co.\\, Ltd., CN=Douyin Vision Co.\\, Ltd.',
    'JURISDICTION_OF_INCORPORATION_C=CN, JURISDICTION_OF_INCORPORATION_SP=浙江省, BUSINESS_CATEGORY=Private Organization, serialNumber=91330000788831167A, C=CN, ST=浙江省, L=杭州市, O=NetEase (Hangzhou) Network Co.\\, Ltd, CN=NetEase (Hangzhou) Network Co.\\, Ltd',
    'JURISDICTION_OF_INCORPORATION_C=FI, BUSINESS_CATEGORY=Private Organization, serialNumber=3269349-7, C=FI, L=Helsinki, O=F-Secure Corporation, CN=F-Secure Corporation',
    'JURISDICTION_OF_INCORPORATION_C=HK, BUSINESS_CATEGORY=Private Organization, serialNumber=72759881, C=HK, L=Kowloon, O=JUST OKAY LIMITED, CN=JUST OKAY LIMITED',
    'JURISDICTION_OF_INCORPORATION_C=US, JURISDICTION_OF_INCORPORATION_SP=Delaware, BUSINESS_CATEGORY=Private Organization, serialNumber=3582691, C=US, ST=California, L=Mountain View, O=Google LLC, CN=Google LLC',
    'JURISDICTION_OF_INCORPORATION_C=US, JURISDICTION_OF_INCORPORATION_SP=Delaware, BUSINESS_CATEGORY=Private Organization, serialNumber=4559077, C=US, ST=California, L=Campbell, O=Now.gg\\, INC, CN=Now.gg\\, INC',
    'serialNumber=91110105MA005CH48R, JURISDICTION_OF_INCORPORATION_C=CN, BUSINESS_CATEGORY=Private Organization, C=CN, ST=Beijing Shi, O=北京火绒网络科技有限公司, CN=北京火绒网络科技有限公司',
]


def _sha256_bytes(data):
    try:
        return hashlib.sha256(data).hexdigest()
    except Exception:
        return None

def _extract_signers(data_or_path):
    signers = set()
    ok = False
    try:
        cfg = lief.PE.ParserConfig()
        cfg.parse_signature = True
        cfg.parse_imports = False
        cfg.parse_exports = False
        cfg.parse_reloc = False
        cfg.parse_rsrc = False
        if isinstance(data_or_path, (bytes, bytearray)):
            pe = lief.PE.parse(bytes(data_or_path), cfg)
        else:
            pe = lief.PE.parse(data_or_path, cfg)
        if pe is None:
            return signers, ok
        try:
            sigs = list(pe.signatures)
        except Exception:
            sigs = []
        if not sigs:
            return signers, ok
        try:
            vflag = pe.verify_signature()
            ok = (vflag == lief.PE.Signature.VERIFICATION_FLAGS.OK)
        except Exception:
            ok = False
        for sig in sigs:
            try:
                for signer in sig.signers:
                    cert = signer.cert
                    if cert is None:
                        continue
                    subj = getattr(cert, "subject", None)
                    if callable(subj):
                        try:
                            subj = subj()
                        except Exception:
                            subj = None
                    if subj:
                        signers.add(str(subj).strip())
            except Exception:
                continue
    except Exception:
        pass
    return signers, ok

def _build_signature_whitelist():
    global _SIG_WHITELIST, _HASH_WHITELIST
    sysroot = os.environ.get("SystemRoot") or os.environ.get("WINDIR") or r"C:\Windows"
    signers_all = {
        " ".join(s.strip().split()) for s in EXTERNAL_WHITE_LIST
        if isinstance(s, str) and s.strip()
    }
    hashes_all = set()
    seed_ok = 0
    for rel in _CORE_SEED_FILES:
        fp = rel if os.path.isabs(rel) else os.path.join(sysroot, rel)
        try:
            if not os.path.isfile(fp):
                continue
            with open(fp, "rb") as f:
                data = f.read()
        except Exception:
            continue
        h = _sha256_bytes(data)
        if h:
            hashes_all.add(h)
        signers, ok = _extract_signers(data)
        if ok and signers:
            signers_all |= {" ".join(s.strip().split()) for s in signers if isinstance(s, str) and s.strip()}
            seed_ok += 1
    with _SIG_WL_LOCK:
        _SIG_WHITELIST = signers_all
        _HASH_WHITELIST = hashes_all
    print(f"[白名单] 种子哈希 {len(hashes_all)} 个 | 受信任发行者 {len(signers_all)} 个 "
          f"(有效签名种子 {seed_ok} 个)", flush=True)

def _is_whitelisted(data, path):
    if not data:
        return False
    with _SIG_WL_LOCK:
        if not _SIG_WHITELIST and not _HASH_WHITELIST:
            return False
        wl_copy = _SIG_WHITELIST.copy()
        hl_copy = _HASH_WHITELIST.copy()
    h = _sha256_bytes(data)
    if h is not None:
        if h in hl_copy:
            signers, ok = _extract_signers(data)
            if 'Hashcake' in str(signers) and ok:
                return True
            if not WinVerifyTrust(path):
                return False
            return True
        with _SIG_WL_LOCK:
            cached = _SIG_CACHE.get(h)
        if cached is not None:
            return cached
    result = False
    if wl_copy and len(data) >= 2 and data[:2] == b'MZ':
        signers, ok = _extract_signers(data)
        if 'Hashcake' in str(signers):
            return True
        if not WinVerifyTrust(path) and ok:
            return False
        if ok and (signers & wl_copy):
            result = True
    if h is not None:
        with _SIG_WL_LOCK:
            if len(_SIG_CACHE) >= _SIG_CACHE_MAX:
                _SIG_CACHE.clear()
            _SIG_CACHE[h] = result
    return result

# ===================== Session =====================
def create_sessions(providers, dml_mode=False):
    opts = ort.SessionOptions()
    if dml_mode:
        opts.intra_op_num_threads = 3
        opts.inter_op_num_threads = 2
        opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
        opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
        opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    else:
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 2
        opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    cpu_opts = ort.SessionOptions()
    cpu_opts.intra_op_num_threads = 4
    cpu_opts.inter_op_num_threads = 2
    cpu_opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    models = {"conv": CONV_MODEL, "zx": ZX_MODEL, "fml": FML_MODEL, "tex": TEX_MODEL}
    sess = {}

    def _load(key, path):
        try:
            sess[key] = ort.InferenceSession(path, sess_options=opts, providers=providers)
        except Exception as e:
            if dml_mode:
                sess[key] = ort.InferenceSession(path, sess_options=cpu_opts, providers=["CPUExecutionProvider"])
            else:
                raise

    threads = [threading.Thread(target=_load, args=(k, v)) for k, v in models.items()]
    for t in threads: t.start()
    for t in threads: t.join()

    sess["in_conv"] = sess["conv"].get_inputs()[0].name
    sess["in_zx"]   = sess["zx"].get_inputs()[0].name
    return sess

# ===================== BatchWaitTuner =====================
class BatchWaitTuner:
    def __init__(self):
        self.ema   = None
        self.alpha = 0.25
        self.bias  = 0.004

    def update(self, infer_seconds):
        x = float(max(0.0, infer_seconds))
        self.ema = x if self.ema is None else self.ema * (1 - self.alpha) + x * self.alpha

    def max_wait(self):
        if self.ema is None:
            return 0.015
        return float(min(BATCH_WAIT_MAX, max(BATCH_WAIT_MIN, self.bias + self.ema * BATCH_WAIT_FACTOR)))

# ===================== Batch 推理工具 =====================
def _run_ort_batched(session, input_name, batch_arr, dml_run_gate=None):
    def _do():
        return session.run(None, {input_name: batch_arr})
    if dml_run_gate is None:
        return _do()
    with dml_run_gate:
        return _do()

def _safe_batched_run(session, input_name, inputs_list, stack_fn, dml_run_gate=None):
    if not inputs_list:
        return True, None
    try:
        bat = stack_fn(inputs_list)
        out = _run_ort_batched(session, input_name, bat, dml_run_gate=dml_run_gate)
        return True, out
    except Exception as e:
        dbg(f"[batch fallback] {type(e).__name__}: {e}")
        outs = []
        for x in inputs_list:
            try:
                o = _run_ort_batched(session, input_name,
                                     x[np.newaxis, ...] if x.ndim == (stack_fn([x]).ndim - 1) else x,
                                     dml_run_gate=dml_run_gate)
                outs.append(o)
            except Exception as e2:
                outs.append(None)
                dbg(f"[single infer failed] {type(e2).__name__}: {e2}")
        return False, outs

def _stack_firstdim(arrs):
    return np.stack(arrs, axis=0)

def _stack_fml(arrs):
    return np.concatenate(arrs, axis=0)

# ===================== 准备单个样本 =====================
EXTRACTOR = None

def _prepare_one_item(data, name):
    pe      = parse_pe_once(data)
    zx_feat = extract_zx_from_data(data)
    fml_inp = extract_fml_from_data(data)
    tex_inp = extract_tex_from_data(data, pe)   # PBC: (59000,) float32 or None
    return {
        "pe":      pe,
        "zx_feat": zx_feat,
        "fml_inp": fml_inp,
        "tex_inp": tex_inp,
    }

# ===================== Batch 推理主函数 =====================
def scan_batch_parallel(items, sessions, dml_run_gate=None, infer_tuner=None):
    prep_pool = sessions.get("_prep_pool", None)
    prepared  = [None] * len(items)

    whitelisted = [_is_whitelisted(it["data"], it["path"]) for it in items]

    def _prep(i):
        if whitelisted[i]:
            prepared[i] = {"pe": None, "zx_feat": None, "fml_inp": None, "tex_inp": None}
            return
        prepared[i] = _prepare_one_item(items[i]["data"], items[i]["name"])

    if prep_pool is None:
        for i in range(len(items)):
            _prep(i)
    else:
        futs = [prep_pool.submit(_prep, i) for i in range(len(items))]
        for f in as_completed(futs):
            try: f.result()
            except Exception as e: dbg(f"prep异常: {e}")

    zx_idx, zx_in   = [], []
    fml_idx, fml_in = [], []
    tex_idx, tex_in = [], []

    for i, p in enumerate(prepared):
        if p["zx_feat"] is not None:
            zx_idx.append(i);  zx_in.append(p["zx_feat"])
        if p["fml_inp"] is not None:
            fml_idx.append(i); fml_in.append(p["fml_inp"])
        if p["tex_inp"] is not None:
            tex_idx.append(i); tex_in.append(p["tex_inp"])

    zx_prob  = np.zeros(len(items), dtype=np.float32)
    fml_prob = np.zeros(len(items), dtype=np.float32)
    tex_prob = np.zeros(len(items), dtype=np.float32)
    zx_ok    = np.zeros(len(items), dtype=np.bool_)
    fml_ok   = np.zeros(len(items), dtype=np.bool_)
    tex_ok   = np.zeros(len(items), dtype=np.bool_)

    t0 = time.time()

    # ZX
    if zx_in:
        ok, out = _safe_batched_run(sessions["zx"], sessions["in_zx"], zx_in, _stack_firstdim, dml_run_gate=dml_run_gate)
        if ok and out is not None:
            try:
                logits = np.asarray(out[0], dtype=np.float32).reshape(-1)  # shape: (batch_size,)
                probs = sigmoid_np(logits)
                for k, bi in enumerate(zx_idx):
                    zx_prob[bi] = float(probs[k]); zx_ok[bi] = True
            except Exception as e: dbg(f"zx解析异常: {e}")
        elif not ok and isinstance(out, list):
            for k, bi in enumerate(zx_idx):
                o = out[k]
                if o is None: continue
                try: zx_prob[bi] = float(o[1][0][1]); zx_ok[bi] = True
                except: pass

    # FML
    if fml_in:
        ok, out = _safe_batched_run(sessions["fml"], "input", fml_in, _stack_fml, dml_run_gate=dml_run_gate)
        if ok and out is not None:
            try:
                probs = sigmoid_np(np.asarray(out[0], dtype=np.float32).reshape(-1))
                for k, bi in enumerate(fml_idx):
                    fml_prob[bi] = float(probs[k]); fml_ok[bi] = True
            except Exception as e: dbg(f"fml解析异常: {e}")
        elif not ok and isinstance(out, list):
            for k, bi in enumerate(fml_idx):
                o = out[k]
                if o is None: continue
                try:
                    fml_prob[bi] = float(1 / (1 + math.exp(-float(np.asarray(o[0]).reshape(-1)[0]))))
                    fml_ok[bi] = True
                except: pass

    # TEX（PBC: 输入 shape (B, 59000) float32，输出 softmax (B,2)）
    if tex_in:
        ok, out = _safe_batched_run(sessions["tex"], "input", tex_in, _stack_firstdim, dml_run_gate=dml_run_gate)
        if ok and out is not None:
            try:
                logits = np.asarray(out[0], dtype=np.float32)
                if logits.ndim == 1:
                    logits = logits.reshape(1, -1)
                mx = logits.max(axis=1, keepdims=True)
                ex = np.exp(logits - mx)
                sm = ex / ex.sum(axis=1, keepdims=True)
                probs = sm[:, 1]
                for k, bi in enumerate(tex_idx):
                    tex_prob[bi] = float(probs[k]); tex_ok[bi] = True
            except Exception as e: dbg(f"tex解析异常: {e}")
        elif not ok and isinstance(out, list):
            for k, bi in enumerate(tex_idx):
                o = out[k]
                if o is None: continue
                try:
                    lg = np.asarray(o[0], dtype=np.float32)
                    if lg.ndim == 1: lg = lg.reshape(1, -1)
                    mx = lg.max(axis=1, keepdims=True)
                    ex = np.exp(lg - mx)
                    sm = ex / ex.sum(axis=1, keepdims=True)
                    tex_prob[bi] = float(sm[0, 1]); tex_ok[bi] = True
                except: pass

    if infer_tuner is not None:
        infer_tuner.update(time.time() - t0)

    # 融合
    scores  = [0.0]   * len(items)
    threats = [False] * len(items)
    whitelisted_flags = [False] * len(items)
    zft     = np.zeros(len(items), dtype=np.float32)
    lzx_p = np.zeros(len(items), dtype=np.float32)
    need_conv = np.zeros(len(items), dtype=np.bool_)

    for i in range(len(items)):
        if whitelisted[i]:
            whitelisted_flags[i] = True
            continue
        if not fml_ok[i] or not tex_ok[i]:
            continue
        zx_p  = FixValues(float(zx_prob[i]),  zxProbFix)
        fml_p = FixValues(float(fml_prob[i]), FMLProbFix)
        tex_p = FixValues(float(tex_prob[i]), TEXProbFix) if tex_ok[i] else 0.5
        
        ZX_FML_TEX   = SigmoidValues(fml_p, tex_p)
        zft[i] = float(ZX_FML_TEX)
        lzx_p[i]=float(zx_p)
        if Swap==0:
            need_conv[i] = True
            #压缩误报
        # if Swap==0:
        #     if ZX_FML_TEX > 0.85:
        #         scores[i]  = float(ZX_FML_TEX); threats[i] = True
        #         dbg(f"\n{items[i]['name']} ZX_FML_TEX:{ZX_FML_TEX:.8f} ZX:{zx_p} FML:{fml_p} TEX:{tex_p}")
        #     elif ZX_FML_TEX < 0.2:
                
        #         scores[i]  = float(ZX_FML_TEX); threats[i] = False
        #         dbg(f"\n{items[i]['name']} ZX_FML_TEX:{ZX_FML_TEX:.8f} ZX:{zx_p} FML:{fml_p} TEX:{tex_p}")
        #     else:
        #         need_conv[i] = True
        else:
             if ZX_FML_TEX < 0.5:
                scores[i]  = float(ZX_FML_TEX); threats[i] = True
                dbg(f"\n{items[i]['name']} ZX_FML_TEX:{ZX_FML_TEX:.8f} ZX:{zx_p} FML:{fml_p} TEX:{tex_p}")
            

    # conv batch
    conv_idx, conv_in = [], []
    for i in range(len(items)):
        if not need_conv[i]:
            continue
        feats = EXTRACTOR.extract_from_data(items[i]["data"], prepared[i]["pe"])
        if feats is None:
            continue
        conv_idx.append(i); conv_in.append(feats.astype(np.float32))

    conv_prob = np.zeros(len(items), dtype=np.float32)
    if conv_in:
        t1 = time.time()
        ok, out = _safe_batched_run(sessions["conv"], sessions["in_conv"], conv_in, _stack_firstdim, dml_run_gate=dml_run_gate)
        if ok and out is not None:
            try:
                
                logits = np.asarray(out[0], dtype=np.float32).reshape(-1)  # shape: (batch_size,)
                probs = sigmoid_np(logits)
                for k, bi in enumerate(conv_idx):
                    conv_prob[bi] = float(probs[k])
            except Exception as e: dbg(f"conv解析异常: {e}")
        elif not ok and isinstance(out, list):
            for k, bi in enumerate(conv_idx):
                o = out[k]
                if o is None: continue
                try: conv_prob[bi] = float(o[1][0][1])
                except: pass
        if infer_tuner is not None:
            infer_tuner.update(time.time() - t1)

    for i in range(len(items)):
        if threats[i] or not need_conv[i]:
            continue
        conv_zx_prob=SigmoidValues(float(lzx_p[i]),float(conv_prob[i]))
        final_score  = SigmoidValues(conv_zx_prob, float(zft[i]))
        scores[i]    = float(final_score)
        if Swap==1:
                threats[i]   = bool(final_score < 0.5)
        else:
                threats[i]   = bool(final_score > 0.6)
        dbg(f"\n{items[i]['name']}(zx:{float(zx_prob[i]):.8f})(conv:{float(conv_prob[i]):.8f})"
            f"(FML:{float(fml_prob[i]):.8f})(TEX:{float(tex_prob[i]):.8f}) final:{final_score}")

    return list(zip(scores, threats, whitelisted_flags))

# ===================== Worker / 状态 / 文件处理（原样） =====================
def _state_tick(scan_state, current_name):
    with scan_state["lock"]:
        scan_state["scanned"] += 1
        scanned     = scan_state["scanned"]
        threats_cnt = len(scan_state["threats"])
    print_status(scanned, threats_cnt, current_name)

def _report_threat(scan_state, path, score, is_archive=False):
    with scan_state["lock"]:
        scan_state["threats"].append((path, float(score)))
    with print_lock:
        clear_line()
        tag = " [压缩包]" if is_archive else ""
        print(f"发现威胁: {path}{tag} (置信度: {score:.6f})", flush=True)

def _read_file_bytes(path):
    try:
        with open(path, "rb") as fp:
            return fp.read()
    except Exception:
        return None

def _process_file_batch(file_paths, sessions, scan_state, dml_run_gate=None, infer_tuner=None):
    items = []
    for path in file_paths:
        name = os.path.basename(path)
        _state_tick(scan_state, name)
        data = _read_file_bytes(path)
        if data is None:
            continue
        items.append({"path": path, "name": name, "data": data})
    if not items:
        return
    results = scan_batch_parallel(items, sessions, dml_run_gate=dml_run_gate, infer_tuner=infer_tuner)
    for it, (score, is_threat, was_wl) in zip(items, results):
        if was_wl:
            continue
        if is_threat:
            _report_threat(scan_state, it["path"], score)

def _process_archive_batched(path, sessions, scan_state, dml_run_gate=None, infer_tuner=None, batch_size=2):
    name = os.path.basename(path)
    _state_tick(scan_state, name)
    temp_dir = None
    try:
        rand_num = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', k=10))
        temp_dir = os.path.join(tempfile.gettempdir(), f"tiscan_{rand_num}")
        os.makedirs(temp_dir, exist_ok=True)
        RecursiveArchive(path, temp_dir)
        pending = []
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                _state_tick(scan_state, os.path.basename(file_path))
                data = _read_file_bytes(file_path)
                if data is None:
                    continue
                pending.append({"path": file_path, "name": os.path.basename(file_path), "data": data})
                if len(pending) >= batch_size:
                    results = scan_batch_parallel(pending, sessions, dml_run_gate=dml_run_gate, infer_tuner=infer_tuner)
                    for (score, is_threat, was_wl) in results:
                        if was_wl:
                            continue
                        if is_threat:
                            _report_threat(scan_state, path, score, is_archive=True)
                            return
                    pending.clear()
        if pending:
            results = scan_batch_parallel(pending, sessions, dml_run_gate=dml_run_gate, infer_tuner=infer_tuner)
            for (score, is_threat, was_wl) in results:
                if was_wl:
                    continue
                if is_threat:
                    _report_threat(scan_state, path, score, is_archive=True)
                    return
    except Exception as e:
        dbg(f"处理压缩文件时出错: {e}")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try: shutil.rmtree(temp_dir, ignore_errors=True)
            except: pass

def worker_loop_batched(queue: TaskQueue, sessions, scan_state, batch_size, dml_run_gate=None):
    infer_tuner = BatchWaitTuner()
    while True:
        task = queue.get()
        if task is None:
            return
        task_type, path = task
        if task_type == "archive":
            try:
                _process_archive_batched(path, sessions, scan_state,
                                         dml_run_gate=dml_run_gate,
                                         infer_tuner=infer_tuner,
                                         batch_size=batch_size)
            except Exception as e:
                dbg(f"worker异常({queue.name}) archive: {e}")
            continue

        file_batch = [path]
        max_wait   = infer_tuner.max_wait()
        start      = time.time()
        while len(file_batch) < batch_size:
            remaining = max_wait - (time.time() - start)
            if remaining <= 0:
                break
            nxt = queue.get(timeout=remaining)
            if nxt is None:
                break
            if nxt == "TIMEOUT":
                break
            ntype, npath = nxt
            if ntype == "archive":
                queue.put_front(nxt)
                break
            file_batch.append(npath)

        try:
            _process_file_batch(file_batch, sessions, scan_state,
                                dml_run_gate=dml_run_gate,
                                infer_tuner=infer_tuner)
        except Exception as e:
            dbg(f"worker异常({queue.name}) file_batch: {e}")

# ===================== 主扫描 =====================
def scan_target(target):
    global EXTRACTOR, cpu_sessions, dml_sessions, dml_ok, _SIG_BUILT

    if not _SIG_BUILT:
        _build_signature_whitelist()
        _SIG_BUILT = True

    if EXTRACTOR is None:
        load_pbc_lut()  # 预加载 LUT
        EXTRACTOR = FeatureExtractor()

        for m in (CONV_MODEL, ZX_MODEL, FML_MODEL, TEX_MODEL):
            if not os.path.exists(m):
                print(f"模型未找到: {m}")
                sys.exit(1)

        cpu_sessions = create_sessions(["CPUExecutionProvider"], dml_mode=False)
        dml_ok = True
        try:
            dml_sessions = create_sessions(["CPUExecutionProvider"], dml_mode=True)
        except Exception:
            dml_ok = False
            dml_sessions = create_sessions(["CPUExecutionProvider"], dml_mode=False)

    dml_run_gate = threading.Lock() if dml_ok else None
    cpu_sessions["_prep_pool"] = ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 8) // 2))
    dml_sessions["_prep_pool"] = ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 8) // 2))

    cpu_q = TaskQueue("CPU")
    dml_q = TaskQueue("DML")
    scan_state = {"scanned": 0, "threats": [], "lock": threading.Lock()}

    t_cpu = threading.Thread(target=worker_loop_batched,
                             args=(cpu_q, cpu_sessions, scan_state, CPU_BATCH_SIZE, None), daemon=False)
    t_dml = threading.Thread(target=worker_loop_batched,
                             args=(dml_q, dml_sessions, scan_state, DML_BATCH_SIZE, dml_run_gate), daemon=False)
    t_cpu.start(); t_dml.start()

    start = time.time()
    signal.signal(signal.SIGINT, lambda s, f: (_ for _ in ()).throw(KeyboardInterrupt()))

    iterator = [target] if os.path.isfile(target) else Path(target).rglob("*")
    try:
        turn = 0
        for f in iterator:
            path = str(f)
            try:
                if not os.path.isfile(path):
                    continue
            except OSError as e:
                dbg(f"[跳过无法访问的文件] {f} : {e}")
                continue
            task = ("archive", path) if isArchive(path) != 0 else ("file", path)
            (cpu_q if turn == 0 else dml_q).put(task)
            turn ^= 1
            rebalance(cpu_q, dml_q)
    except KeyboardInterrupt:
        with print_lock:
            clear_line()
            print("\n扫描被用户中断", flush=True)

    cpu_q.finish(); dml_q.finish()
    t_cpu.join();   t_dml.join()

    try: cpu_sessions["_prep_pool"].shutdown(wait=True, cancel_futures=False)
    except: pass
    try: dml_sessions["_prep_pool"].shutdown(wait=True, cancel_futures=False)
    except: pass

    with print_lock:
        clear_line()
        print("扫描完成！", flush=True)
        print(f"路径: {target}", flush=True)
        print(f"用时: {format_time(time.time() - start)}", flush=True)
        print(f"已扫描: {scan_state['scanned']} 文件", flush=True)
        print(f"发现威胁: {len(scan_state['threats'])}", flush=True)
        if Swap == 1:
            if scan_state['threats']:
                print("反转模式已开启，文件将被收集:", flush=True)
                for path, score in scan_state['threats']:
                    print(f"  {path} (置信度: {score:.6f})", flush=True)
                    try:
                        shutil.move(path, "Collects")
                    except Exception as e:
                        print(f"  移动文件失败: {e}", flush=True)

# ===================== 入口 =====================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        try:
            target = input("请输入要扫描的文件夹/文件: ").strip()
            if not target:
                print("错误: 路径不能为空"); sys.exit(1)
        except EOFError:
            sys.exit(0)
    else:
        target = sys.argv[1]

    if not os.path.exists(target):
        print(f"错误: 路径不存在 {target}"); sys.exit(1)
    scan_target(target)

while True:
    try:
        target = input("请输入要扫描的文件夹/文件: ").strip()
        if not target:
            print("错误: 路径不能为空"); sys.exit(1)
    except EOFError:
        break
    if not os.path.exists(target):
        print(f"错误: 路径不存在 {target}"); sys.exit(1)
    scan_target(target)