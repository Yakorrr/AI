class PriorityQueue:
    def __init__(self):
        self.items = []

    def __repr__(self):
        return f'PriorityQueue(items={self.items})'

    def __str__(self):
        return ' '.join([str(i) for i in self.items])

    def push(self, item):
        self.items.append(item)

    def pop(self):
        try:
            index = 0

            for i in range(len(self.items)):
                if self.items[i][0] > self.items[index][0]: index = i

            item = self.items[index]
            del self.items[index]

            return item
        except IndexError:
            print('ERROR! Trying to pop an element from an empty priority queue.')

    def length(self):
        return len(self.items)

    def isEmpty(self):
        return self.length() == 0
