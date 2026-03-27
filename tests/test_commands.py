def test_parse_command_switch():
    from cli.commands import parse_command
    cmd = parse_command("/switch review")
    assert cmd.name == "switch"
    assert cmd.args == "review"


def test_parse_command_no_args():
    from cli.commands import parse_command
    cmd = parse_command("/agents")
    assert cmd.name == "agents"
    assert cmd.args == ""


def test_parse_command_resume():
    from cli.commands import parse_command
    cmd = parse_command("/resume thread-abc")
    assert cmd.name == "resume"
    assert cmd.args == "thread-abc"


def test_parse_non_command_returns_none():
    from cli.commands import parse_command
    assert parse_command("hello world") is None
    assert parse_command("") is None


def test_parse_command_exit():
    from cli.commands import parse_command
    cmd = parse_command("/exit")
    assert cmd.name == "exit"


def test_parse_command_help():
    from cli.commands import parse_command
    cmd = parse_command("/help")
    assert cmd.name == "help"
