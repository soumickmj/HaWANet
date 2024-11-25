from pathlib import Path
import pandas as pd
import tempfile
from shutil import copyfile


class Database:
    def __init__(self, dir_path, recursive):
        self.dir_path = Path(dir_path)
        self.recursive = recursive
        self.db_path = self.dir_path.joinpath('tags.csv')
        self.db = self.load()

    def load(self):
        if not self.db_path.is_file():
            return {}

        df = pd.read_csv(self.db_path, index_col=0)
        df.tags = df.tags.fillna('')
        df.tags = df.tags.apply(lambda x: x.split(';'))
        df.tags = df.tags.apply(lambda x: [] if all(v == '' for v in x) else x)
        data = df.to_dict(orient='index')
        return data

    def save(self):
        df = pd.DataFrame(self.db).T
        df.tags = df.tags.apply(lambda x: ';'.join(x))
        prefix = self.dir_path.stem if self.dir_path.stem != 'visualized' else self.dir_path.parent.stem
        with tempfile.NamedTemporaryFile('w+t', delete=False, prefix=f'{prefix}_') as cf:
            df.to_csv(cf)
            temp_file_path = cf.name
            print(f'Temporary checkpoint at {cf.name}')
        copyfile(temp_file_path, self.db_path)

    def add_record(self, filename, selection, tags: list):
        filename = Path(filename)
        name = filename.name if not self.recursive else f'{filename.parent.name}/{filename.name}'
        record = {name: {'selection': selection, 'tags': tags}}
        msg = 'Added record'
        if filename.name in self.db:
            msg = 'Updated record'
        self.db.update(record)
        return msg

    def __getitem__(self, item):
        filename = Path(item)
        name = filename.name if not self.recursive else f'{filename.parent.name}/{filename.name}'
        return self.db.get(name, None)

    def filter(self, selection: list = None, tags: list = None):
        if all([selection is None, tags is None]):
            return {k: True for k in self.db.keys()}

        df = pd.DataFrame(self.db).T
        mask = [True] * len(df)
        if selection is not None and len(selection):
            mask &= df.selection.apply(lambda x: any(t == x for t in selection))
        if tags is not None and len(tags):
            mask &= df.tags.apply(lambda x: any(t in y for t in tags for y in x))

        filtered_urls = df.index[mask].tolist()
        return {k: True if k in filtered_urls else False for k in self.db.keys()}
