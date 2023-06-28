import heapq


class PriorityQueue:
    def __init__(self, items=None):
        self._items = []

        if items:
            for priority, item in items:
                self.push(item, priority)

    def push(self, item, priority):
        entry = (priority, item)
        heapq.heappush(self._items, entry)

    def pop(self):
        try:
            _, item = heapq.heappop(self._items)
            return item
        except IndexError:
            print('ERROR! Trying to pop an element from an empty priority queue.')

    def length(self):
        return len(self._items)

    def get_items(self):
        return self._items

    def is_empty(self):
        return self.length() == 0

    def __repr__(self):
        return f'PriorityQueue(items={self._items})'

    def __str__(self):
        res = '['
        for priority, _, item in self._items:
            res += f' {item}({priority}) '
        res += ']'
        return res
