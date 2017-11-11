def probe_operation(soperation, domain, nprobes,
                    random_type, dtype):

    for i in range(nprobes):
        f = Field.from_random(random_type=random_type, domain=domain,
                              dtype=dtype)
        tmp = operator(f)
        if i==0:
            mean = [0]*len(tmp)
            var = [0]*len(tmp)
        for i in range(len(tmp)):
            mean[i] += tmp[i]
            var[i] += tmp[i]**2
    for i in range(len(tmp)):
        mean[i] *= 1./nprobes
        var[i] *= 1./nprobes
        var[i] -= mean[i]**2
    return mean, var
