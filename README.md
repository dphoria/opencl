# Install Driver

- CPU: Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz

- VGA:
```json
{
    "id" : "display",
    "class" : "display",
    "claimed" : true,
    "handle" : "PCI:0000:00:02.0",
    "description" : "VGA compatible controller",
    "product" : "UHD Graphics",
    "vendor" : "Intel Corporation",
    "physid" : "2",
    "businfo" : "pci@0000:00:02.0",
    "logicalname" : "/dev/fb0",
    "dev" : "29:0",
    "version" : "02",
    "width" : 64,
    "clock" : 33000000,
    "configuration" : {
        "depth" : "32",
        "driver" : "i915",
        "latency" : "0",
        "mode" : "3840x2160",
        "visual" : "truecolor",
        "xres" : "3840",
        "yres" : "2160"
    },
    "capabilities" : {
        "pciexpress" : "PCI Express",
        "msi" : "Message Signalled Interrupts",
        "pm" : "Power Management",
        "vga_controller" : true,
        "bus_master" : "bus mastering",
        "cap_list" : "PCI capabilities listing",
        "rom" : "extension ROM",
        "fb" : true
    }
}
```
- OS:  
5.11.0-38-generic #42~20.04.1-Ubuntu SMP Tue Sep 28 20:41:07 UTC 2021 x86_64 x86_64 x86_64 GNU/Linux

- OpenCL Driver for Intel UHD Graphics:  
https://github.com/intel/compute-runtime/blob/master/opencl/doc/DISTRIBUTIONS.md  
```Shell
apt install intel-opencl-icd
```

---

## Verify Driver ##

CMake (`3.16.3-1ubuntu1`) found none.

```Shell
-- Looking for CL_VERSION_2_2
-- Looking for CL_VERSION_2_2 - not found
-- Looking for CL_VERSION_2_1
-- Looking for CL_VERSION_2_1 - not found
-- Looking for CL_VERSION_2_0
-- Looking for CL_VERSION_2_0 - not found
-- Looking for CL_VERSION_1_2
-- Looking for CL_VERSION_1_2 - not found
-- Looking for CL_VERSION_1_1
-- Looking for CL_VERSION_1_1 - not found
-- Looking for CL_VERSION_1_0
-- Looking for CL_VERSION_1_0 - not found
```

One file that `FindOpenCL.cmake` looks for in addition to the driver `*.so` file is `cl.h`. Search hits for packages that install `cl.h` on https://packages.ubuntu.com/ include `opencl-c-headers`.  

```Shell
apt install opencl-c-headers
```

CMake does better, but still unresolved.

```Shell
-- Looking for CL_VERSION_2_2
-- Looking for CL_VERSION_2_2 - found
CMake Error at /usr/share/cmake-3.16/Modules/FindPackageHandleStandardArgs.cmake:146 (message):
  Could NOT find OpenCL (missing: OpenCL_LIBRARY) (found version "2.2")
Call Stack (most recent call first):
  /usr/share/cmake-3.16/Modules/FindPackageHandleStandardArgs.cmake:393 (_FPHSA_FAILURE_MESSAGE)
  /usr/share/cmake-3.16/Modules/FindOpenCL.cmake:150 (find_package_handle_standard_args)
  CMakeLists.txt:4 (find_package)
```

Further reading `FindOpenCL.cmake` shows

```CMake
find_library(OpenCL_LIBRARY
  ...
  PATHS
    ENV AMDAPPSDKROOT
    ENV CUDA_PATH
  ...
)
```

Neither `AMDAPPSDKROOT` nor `CUDA_PATH` environment variables has `libOpenCL.so`.

```Shell
$ dpkg -L ocl-icd-libopencl1 | grep -E '\.so'
/usr/lib/x86_64-linux-gnu/libOpenCL.so.1.0.0
/usr/lib/x86_64-linux-gnu/libOpenCL.so.1
/usr/share/man/man7/libOpenCL.so.7.gz
```

Modified CMake file to add to search paths prior to `find_package()`.

```CMake
set(OpenCL_ROOT /usr/lib/x86_64-linux-gnu/)
```

And made symlink so `find_library(OpenCL)` can find `libOpenCL.so`.

```Shell
cd /usr/lib/x86_64-linux-gnu
ln -s libOpenCL.so.1 libOpenCL.so
```

Now CMake finds it.

```Shell
-- Looking for CL_VERSION_2_2
-- Looking for CL_VERSION_2_2 - found
-- Found OpenCL: /usr/lib/x86_64-linux-gnu/libOpenCL.so (found version "2.2") 
```

Also confiremd by `clinfo`

```Shell
$ apt install clinfo
$ clinfo -a
...
  Platform Name                                   Intel(R) OpenCL HD Graphics
Number of devices                                 1
  Device Name                                     Intel(R) Gen9 HD Graphics NEO
...
```
