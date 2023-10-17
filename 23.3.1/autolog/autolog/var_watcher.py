from IPython import InteractiveShell

class VarWatcher(object):
    def __init__(self, ip: InteractiveShell):
        self.shell = ip
        self.last_x = None
        self.local_vars = None

    def pre_execute(self):
        self.last_x = self.shell.user_ns.get('x', None)

    def pre_run_cell(self, info):
        print('info.raw_cell =', info.raw_cell)
        print('info.store_history =', info.store_history)
        print('info.silent =', info.silent)
        print('info.shell_futures =', info.shell_futures)
        print('info.cell_id =', info.cell_id)
        print(dir(info))

    def post_execute(self):
        self.local_vars = locals()
        print(self.local_vars)
        # if self.shell.user_ns.get('x', None) != self.last_x:
        #     print("x changed!")

    def post_run_cell(self, result):
        # print('result = ', _identify_assets(_get_local_variables()))
        # self.shell.events.unregister("post_execute", self.post_execute)
        print("post_run_cell")
        self.local_vars = locals()
        print(self.local_vars.keys())


def load_ipython_extension(ip):
    from .autolog import autolog
    vw = VarWatcher(ip)
    # ip.events.register('pre_execute', vw.pre_execute)
    # ip.events.register('pre_run_cell', vw.pre_run_cell)
    ip.events.register('post_run_cell', vw.post_run_cell)
    ip.events.register('post_run_cell', autolog)
    # ip.events.register('post_run_cell', vw.post_run_cell)
    return vw