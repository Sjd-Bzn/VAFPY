

#@pytest.mark.skip
def test_diamond():
    import afqmc_RUN_kpts 
    with open("outcar.txt", "r") as current:
        current_lines = current.readlines()
    with open("outcar_ref.txt", "r") as reference:
        reference_lines = reference.readlines()
    assert current_lines == reference_lines
