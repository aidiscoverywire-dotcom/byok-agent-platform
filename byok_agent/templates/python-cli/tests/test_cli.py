from cli import main


def test_main_ok(capsys):
    assert main(["--name", "tester"]) == 0
    assert "hello, tester" in capsys.readouterr().out
