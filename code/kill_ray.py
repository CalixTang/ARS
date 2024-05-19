import ray

if ray.is_initialized():
    ray.shutdown()
    print('ray shutdown')
