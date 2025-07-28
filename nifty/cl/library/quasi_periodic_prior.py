import nifty.cl as ift
from typing import Union


def build_prior_operator(domain_key, values):
    values = _transform_setting(values)

    distribution = values.get('distribution')

    if distribution in ['uniform']:
        operator = ift.UniformOperator(
            ift.UnstructuredDomain(values['N_copies']),
            loc=values['mean'],
            scale=values['sigma']
        ).ducktape(domain_key).ducktape_left(domain_key)

    elif distribution in ['normal']:
        operator = ift.NormalTransform(
            mean=values['mean'],
            sigma=values['sigma'],
            key=domain_key,
            N_copies=values['N_copies']
        ).ducktape_left(domain_key)

    elif distribution in ['log_normal', 'lognormal']:
        operator = ift.LognormalTransform(
            mean=values['mean'],
            sigma=values['sigma'],
            key=domain_key,
            N_copies=values['N_copies']
        ).ducktape_left(domain_key)

    elif distribution is None:
        value = ift.Field.from_raw(
            ift.UnstructuredDomain(values['N_copies']), float(values['mean'])
        )
        print(f'Constant ({domain_key}): {value.val}')
        operator = {domain_key: value.val}

    else:
        print('This distribution is not implemented')
        raise NotImplementedError

    return operator


def _transform_setting(values: Union[dict, tuple]):
    if isinstance(values, dict):
        return values

    return dict(
        distribution=values[0],
        mean=values[1],
        sigma=values[2]
    )
