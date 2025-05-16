#@pytest.mark.skip
import os
import pathlib
import pytest
# from vafpy.run_kpts import main

@pytest.mark.parametrize("testcase", ["diamond"])
def test_regression(testcase, tmp_path, monkeypatch):
    # Create symbolic link to all files in reference directory
    reference_directory = pathlib.Path(__file__).parent / testcase
    print(reference_directory)
    for file in reference_directory.iterdir():
        (tmp_path / file.name).symlink_to(file.resolve())

    # Change working directory to tmp_path for the duration of the test
    monkeypatch.chdir(tmp_path)

    for file in tmp_path.iterdir():
        print(file)
    assert False
#     os.environ["REGRESSION_TEST"] = "1"
#     os.chdir("test/diamond")
#     print(pathlib.Path.cwd())
#
#     # First run: generate outcar.txt and move to reference
# #    subprocess.run(["python", "afqmc_RUN_kpts.py"], check=True)
# #    os.rename("outcar.txt", "outcar_ref.txt")
#
#     # Second run: generate new outcar.txt
# #    subprocess.run(["python", "afqmc_RUN_kpts.py"], check=True)
#     main()
#
#     # Compare output files
#     with open("outcar.txt", "r") as current, open("outcar_ref.txt", "r") as reference:
#         current_lines = current.readlines()
#         reference_lines = reference.readlines()
#
#     assert current_lines == reference_lines, "Outputs do not match!"
#
#     os.chdir("../..")
