import json
import os
import subprocess

import bm


class Startup(bm.Scenario):
    import_ddtrace: bool
    import_ddtrace_auto: bool
    import_flask: bool
    import_django: bool
    runtime_metrics_enabled: bool
    appsec_enabled: bool
    profiling_enabled: bool
    env: str

    def run(self):
        env = os.environ.copy()
        if self.env:
            parsed_env: dict[str, str] = json.loads(self.env)
            env.update(parsed_env)

        if self.runtime_metrics_enabled:
            env["DD_RUNTIME_METRICS_ENABLED"] = "1"
        if self.appsec_enabled:
            env["DD_APPSEC_ENABLED"] = "1"

        commands: list[str] = []
        if self.import_ddtrace:
            commands.append("import ddtrace")
        if self.import_ddtrace_auto:
            commands.append("import ddtrace.auto")
        if self.profiling_enabled:
            commands.append("import ddtrace.profiling.auto")

        # Always do this last, we want to import/patch ddtrace first
        # DEV: We don't need to create/start an app, we just want to see the impact on import time
        #      with and without patching. We have other benchmarks to test overhead of requests.
        if self.import_flask:
            commands.append("import flask")
        if self.import_django:
            # `import django` doesn't really do anything, `django.core.management` is what `manage.py` uses
            commands.append("import django.core.management")

        args = ["python", "-c"] + [";".join(commands)]

        def _(loops: int):
            for _ in range(loops):
                subprocess.check_call(args=args, env=env)

        yield _
