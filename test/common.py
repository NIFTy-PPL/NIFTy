from nose_parameterized import parameterized


def custom_name_func(testcase_func, param_num, param):
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in param.args)),
    )

expand = lambda z: parameterized.expand(z,
                                        testcase_func_name=custom_name_func)
