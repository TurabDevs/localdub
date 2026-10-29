"""Build llama.cpp (CUDA, Tesla T4 = sm_75) once on Kaggle and keep the binaries as this kernel's output,
so the Muse Glimmer kernels can attach them instead of compiling every run.
Muse Glimmer needs llama.cpp build b10353 or newer (architecture merged 2026-08-10)."""
import glob, os, subprocess, time
W = "/kaggle/working"; t0 = time.time()
def sh(c):
    print("+", c, flush=True); r = subprocess.run(c, shell=True); print(f"  rc={r.returncode} t={time.time()-t0:.0f}s", flush=True); return r.returncode
sh("nvidia-smi || true; which nvcc || ls /usr/local/cuda*/bin/nvcc")
sh("export PATH=$PATH:/usr/local/cuda/bin; nvcc --version | tail -2")
sh("cd /tmp && git clone --depth 1 https://github.com/ggml-org/llama.cpp")
sh("cd /tmp/llama.cpp && git log -1 --format='%H %cd' && grep -c LLM_ARCH_MUSE_GLIMMER src/llama-arch.cpp")
rc = sh("cd /tmp/llama.cpp && export PATH=$PATH:/usr/local/cuda/bin && cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON "
        "-DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/tmp/cmake.log 2>&1; tail -20 /tmp/cmake.log; "
        f"cmake --build build --config Release -j{os.cpu_count()} --target llama-server llama-mtmd-cli llama-cli 2>&1 | tail -15")
os.makedirs(f"{W}/bin", exist_ok=True)
sh(f"cp /tmp/llama.cpp/build/bin/llama-server /tmp/llama.cpp/build/bin/llama-mtmd-cli /tmp/llama.cpp/build/bin/llama-cli {W}/bin/ ; ls -la {W}/bin; {W}/bin/llama-server --version")
open(f"{W}/BUILD_OK" if os.path.exists(f"{W}/bin/llama-server") else f"{W}/BUILD_FAILED", "w").write(str(time.time() - t0))
