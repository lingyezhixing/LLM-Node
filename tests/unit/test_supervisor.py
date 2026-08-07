import asyncio
import sys

from llm_node.supervisor import ProcessRecord, ProcessRunner, Supervisor


def test_supervisor_implements_process_runner():
    sup = Supervisor()
    assert isinstance(sup, ProcessRunner)


def test_spawn_returns_process_record_and_exits():
    async def main():
        sup = Supervisor()
        cmd = [sys.executable, "-c", "print('hi')"]
        rec = await sup.spawn(cmd, shell=False)
        assert isinstance(rec, ProcessRecord)
        assert rec.pid > 0
        await asyncio.sleep(1.0)

    asyncio.run(main())


def test_alive_unknown_pid_is_false():
    sup = Supervisor()
    assert sup.alive(99999999) is False


def test_on_exit_callback_fires_when_process_exits():
    async def main():
        sup = Supervisor()
        seen = []
        sup.on_exit(0, lambda code: seen.append(code))
        cmd = [sys.executable, "-c", "print('hi')"]
        rec = await sup.spawn(cmd, shell=False)
        sup.on_exit(rec.pid, lambda code: seen.append(code))
        await asyncio.sleep(1.0)
        assert seen, "on_exit callback did not fire"
        assert seen[-1] in (0, None)

    asyncio.run(main())


def test_kill_tree_clears_process_tables():
    async def main():
        sup = Supervisor()
        rec = await sup.spawn([sys.executable, "-c", "import time; time.sleep(30)"], shell=False)
        sup.on_exit(rec.pid, lambda code: None)
        await asyncio.sleep(0.3)
        assert rec.pid in sup._procs and rec.pid in sup._wait_tasks and rec.pid in sup._exit_cbs
        await sup.kill_tree(rec.pid)
        assert rec.pid not in sup._procs
        assert rec.pid not in sup._exit_cbs
        await asyncio.sleep(0.5)
        assert rec.pid not in sup._wait_tasks

    asyncio.run(main())


def test_spawn_captures_stdout_and_stderr_via_on_output():
    received = []

    async def go():
        sup = Supervisor()

        def on_output(line, stream):
            received.append((line, stream))

        cmd = 'python -c "import sys; print(\\"out-line\\"); sys.stderr.write(\\"err-line\\" + chr(10)); sys.stderr.flush()"'
        rec = await sup.spawn(cmd, on_output=on_output)
        await sup._wait_tasks[rec.pid]
        await asyncio.sleep(0.05)

    asyncio.run(go())
    assert ("out-line", "out") in received
    assert ("err-line", "err") in received


def test_natural_exit_cleans_all_tables():
    async def main():
        sup = Supervisor()
        exited = asyncio.Event()
        proc = await sup.spawn([sys.executable, "-c", "pass"], on_output=lambda _line, _s: None)
        assert proc.pid in sup._readers
        assert proc.pid in sup._procs
        sup.on_exit(proc.pid, lambda _rc: exited.set())
        await asyncio.wait_for(exited.wait(), timeout=5)
        for _ in range(100):
            if not sup._procs and not sup._readers and not sup._wait_tasks:
                break
            await asyncio.sleep(0.02)
        assert sup._procs == {}
        assert sup._exit_cbs == {}
        assert sup._readers == {}
        assert sup._wait_tasks == {}

    asyncio.run(main())


def test_kill_cleans_all_tables():
    async def main():
        sup = Supervisor()
        proc = await sup.spawn(
            [sys.executable, "-c", "import time; time.sleep(60)"], on_output=lambda _line, _s: None
        )
        assert proc.pid in sup._readers
        assert proc.pid in sup._procs
        assert await sup.kill_tree(proc.pid)
        assert sup._procs == {}
        assert sup._exit_cbs == {}
        assert sup._readers == {}

    asyncio.run(main())


def test_fast_kill_does_not_leak_wait_tasks():
    async def main():
        sup = Supervisor()
        for _ in range(3):
            proc = await sup.spawn([sys.executable, "-c", "import time; time.sleep(60)"])
            assert await sup.kill_tree(proc.pid)
        for _ in range(100):
            if not sup._wait_tasks:
                break
            await asyncio.sleep(0.02)
        assert sup._wait_tasks == {}
        assert sup._procs == {}
        assert sup._readers == {}

    asyncio.run(main())
