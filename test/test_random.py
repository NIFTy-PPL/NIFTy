import numpy as np

import nifty6 as ift


def test_rand1():
    ift.random.push_sseq_from_seed(31)
    a = ift.random.current_rng().integers(0,1000000000)
    ift.random.pop_sseq()
    ift.random.push_sseq_from_seed(31)
    b = ift.random.current_rng().integers(0,1000000000)
    np.testing.assert_equal(a,b)


def test_rand2():
    ift.random.push_sseq_from_seed(31)
    sseq = ift.random.spawn_sseq(10)
    ift.random.push_sseq(sseq[2])
    a = ift.random.current_rng().integers(0,1000000000)
    ift.random.pop_sseq()
    ift.random.push_sseq(sseq[2])
    b = ift.random.current_rng().integers(0,1000000000)
    ift.random.pop_sseq()
    np.testing.assert_equal(a,b)
    ift.random.pop_sseq()


def test_rand3():
    ift.random.push_sseq_from_seed(31)
    sseq = ift.random.spawn_sseq(10)
    ift.random.push_sseq(sseq[2])
    a = ift.random.current_rng().integers(0,1000000000)
    ift.random.pop_sseq()
    ift.random.pop_sseq()
    ift.random.push_sseq_from_seed(31)
    sseq = ift.random.spawn_sseq(1)
    sseq = ift.random.spawn_sseq(1)
    sseq = ift.random.spawn_sseq(1)
    ift.random.push_sseq(sseq[0])
    b = ift.random.current_rng().integers(0,1000000000)
    ift.random.pop_sseq()
    np.testing.assert_equal(a,b)
    ift.random.pop_sseq()


def test_rand4():
    ift.random.push_sseq_from_seed(31)
    a = ift.random.current_rng().integers(0,1000000000)
    ift.random.push_sseq_from_seed(31)
    b = ift.random.current_rng().integers(0,1000000000)
    ift.random.pop_sseq()
    ift.random.pop_sseq()
    np.testing.assert_equal(a,b)


def test_rand5():
    ift.random.push_sseq_from_seed(31)
    a = ift.random.current_rng().integers(0,1000000000)
    ift.random.push_sseq_from_seed(31)
    b = ift.random.current_rng().integers(0,1000000000)
    c = ift.random.current_rng().integers(0,1000000000)
    ift.random.pop_sseq()
    d = ift.random.current_rng().integers(0,1000000000)
    ift.random.pop_sseq()
    np.testing.assert_equal(a,b)
    np.testing.assert_equal(c,d)
