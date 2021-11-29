#include <iostream>

template<typename ... Args>
auto d_ocl::utils::checkRun(std::function<cl_int(Args ...)> openclFunc, Args&& ... args) -> bool
{
    if (!openclFunc) {
        std::cerr << "empty opencl api function passed into checkRun()" << std::endl;
        return false;
    }

    cl_int status = openclFunc(std::forward<Args>(args) ...);
    if (status != CL_SUCCESS) {
        // some_function() -> CL_SOME_ERROR(-n)
        std::cerr << openclFunc.target_type().name() << "() -> " << errorString(status) << "(" << status << ")" << std::endl;
        return false;
    }

    return true;
}

template<typename T>
auto d_ocl::utils::manager<T>::makeShared(T openclObject, opencl_release_func releaseFunc)
    -> std::shared_ptr<manager<T>>
{
    if (openclObject == nullptr) {
        // operator bool will fail test
        return std::shared_ptr<manager<T>>();
    }

    return std::make_shared<manager<T>>(openclObject, releaseFunc);
}

template<typename T>
d_ocl::utils::manager<T>::manager(T openclObject, opencl_release_func releaseFunc)
{
    this->openclObject = openclObject;
    this->releaseFunc = releaseFunc;
}

template<typename T>
d_ocl::utils::manager<T>::~manager()
{
    if (openclObject != nullptr && releaseFunc != nullptr) {
        releaseFunc(openclObject);
    }
}
