import pytest
import sys
sys.path.append('/home/caio/Projects/CarbonateDeposition/Repositories/psd-simulations-msm/vessel_carbonate/')
import calciumcarbonate_supersaturation_module as carbonate_eq
import calciumcarbonate_equilibrium
import numba
import numpy as np

tuple_cC_cCa_cNa_cCl =  (0.014148761431388462, 0.0002063963961295059, 0.0141497213599971, 0.00041471264947629455)
solution_output = {
    'CO2': -3.7436989734580055,
    'CO3--': -3.8463381366427174,
    'Ca++': -3.773955242862029,
    'CaCO3(aq)': -4.892726948026751,
    'CaHCO3+': -4.588339296697599,
    'CaOH+': -8.438640323239587,
    'Cl-': -3.3822527176768054,
    'H+': -8.136083887818444,
    'HCO3-': -1.8639030391483653,
    'Na+': -1.8525625558895142,
    'NaCO3-': -4.639381707219435,
    'NaHCO3': -4.073177641529078,
    'NaOH': -7.896478668071072,
    'OH-': -5.754204065690355
}
solution_array = np.array([-8.13552671, -5.75476565, -3.74315   , -3.84691647, -1.86391565, -3.77496305, -8.44020117, -4.58935117, -4.89429586, -1.85256158, -7.89703487, -4.63995052, -4.07318487])
S = 2.7363745379475772

def test_solve_for_carboante_equilibrium():
    eq = carbonate_eq.CalciumCarbonateReaction()
    eq.solve(*tuple_cC_cCa_cNa_cCl)


    assert np.isclose(eq.S, S)
    assert np.isclose(np.sum(eq.sol['x'] - solution_array), 0.0, atol = 1e-6)

def test_residual():
    x = calciumcarbonate_equilibrium.x_guess
    calc = calciumcarbonate_equilibrium.sodiumBicarbonateMixture_residual_function(
        x, *tuple_cC_cCa_cNa_cCl
    )

    assert np.isclose(calc[0], 6.149394151824478)
    assert np.isclose(calc[1], 11.193322127000327)

def test_solve_new():
    x_guess = calciumcarbonate_equilibrium.x_guess
    x, pH, sigma, Ksp, S_calc, IAP, I = calciumcarbonate_equilibrium.solve(*tuple_cC_cCa_cNa_cCl, x_guess)

    print(x - solution_array)
    print('S calc = {} and S expected = {}'.format(S_calc, S))
    assert np.isclose(np.sum(x - solution_array), 0.0, atol = 1e-6)
    assert np.isclose(S_calc, S)


if __name__ == "__main__":
    # test_solve_for_carboante_equilibrium()
    test_solve_new()
    # test_residual()
