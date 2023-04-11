class Stack:
    def __init__(self, items=None):
        self._items = []

        if items:
            for item in items:
                self.push(item)

    def push(self, item):
        self._items.append(item)

    def pop(self):
        try:
            return self._items.pop()
        except IndexError:
            print('ERROR! Trying to pop an element from an empty stack.')

    def length(self):
        return len(self._items)

    def is_empty(self):
        return self.length() == 0

    def __repr__(self):
        return f'Stack(items={self._items})'

    def __str__(self):
        return f"[{', '.join(self._items)}]"
