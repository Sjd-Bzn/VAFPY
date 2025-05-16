

#@pytest.mark.skip
import os
import pathlib
from vafpy.run_kpts import main

def test_diamond():
    os.environ["REGRESSION_TEST"] = "1"
    os.chdir("test/diamond")
    print(pathlib.Path.cwd())

    # First run: generate outcar.txt and move to reference
#    subprocess.run(["python", "afqmc_RUN_kpts.py"], check=True)
#    os.rename("outcar.txt", "outcar_ref.txt")

    # Second run: generate new outcar.txt
#    subprocess.run(["python", "afqmc_RUN_kpts.py"], check=True)
    main()

    # Compare output files
    with open("outcar.txt", "r") as current, open("outcar_ref.txt", "r") as reference:
        current_lines = current.readlines()
        reference_lines = reference.readlines()

    assert current_lines == reference_lines, "Outputs do not match!"

    os.chdir("../..")
