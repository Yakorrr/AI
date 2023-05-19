class Queue:
    def __init__(self, items=None):
        self._items = []

        if items:
            for item in items:
                self.push(item)

    def push(self, item):
        self._items.append(item)

    def pop(self):
        try:
            item = self._items[0]
            self._items = self._items[1:]
            return item
        except IndexError:
            print('ERROR! Trying to pop an element from an empty queue.')

    def length(self):
        return len(self._items)

    def is_empty(self):
        return len(self._items) == 0

    def __repr__(self):
        return f'Queue(items={self._items})'

    def __str__(self):
        return f"[{', '.join(self._items)}]"
