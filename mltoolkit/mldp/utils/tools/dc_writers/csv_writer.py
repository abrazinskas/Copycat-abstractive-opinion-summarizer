from .base_writer import BaseWriter


class CsvWriter(BaseWriter):

    def __init__(self, f, repr_funcs=None, sep='\t'):
        super(CsvWriter, self).__init__(f=f, repr_funcs=repr_funcs)
        self.sep = sep

    def _exit(self, exc_type, exc_val, exc_tb):
        pass

    def _write(self, data_chunk):
        if self.f.tell() == 0:
            self.f.write(self.sep.join(data_chunk.keys()) + "\n")
        for indx in range(len(data_chunk)):
            strs = []
            for fn, fv in data_chunk.items():
                repr_func = self.get_repr_func(fn)
                cfv = fv[indx]
                strs.append(repr_func(cfv))
            self.f.write(self.sep.join(strs) + "\n")

    def get_repr_func(self, fname):
        if fname not in self.repr_funcs:
            return str
        return self.repr_funcs[fname]
