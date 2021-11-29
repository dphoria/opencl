template<typename T>
using opencl_release_func = int (*)(T);

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
        (*releaseFunc)(openclObject);
    }
}
