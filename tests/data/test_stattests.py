import cmdstanpy
from cmdstanpy import CmdStanModel
import tempfile
import os

from verona.evaluation.stattests.stan_codes import STAN_CODE


def test_compile_stan_models():
    with tempfile.NamedTemporaryFile(suffix='.stan', delete=False) as temp:
        temp.write(STAN_CODE.HIERARCHICAL_TEST.encode('utf-8'))
        temp_file_name = temp.name  # Save the filename to use later

    model = CmdStanModel(stan_file=temp_file_name)
    assert model is not None
    assert isinstance(model, cmdstanpy.model.CmdStanModel)
    os.remove(temp_file_name)

    with tempfile.NamedTemporaryFile(suffix='.stan', delete=False) as temp:
        temp.write(STAN_CODE.PLACKETT_LUCE_TEST_V3.encode('utf-8'))
        temp_file_name = temp.name  # Save the filename to use later

    model = CmdStanModel(stan_file=temp_file_name)
    assert model is not None
    assert isinstance(model, cmdstanpy.model.CmdStanModel)
    os.remove(temp_file_name)
