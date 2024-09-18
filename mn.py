def counter(arr):
    for i in arr:
        yield i

arr = [1,2,3,4,5]
c = counter(arr)
for i in c:
    print(i)