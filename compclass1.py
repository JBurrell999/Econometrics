#lol this is the introduction to python
myName = None
a = 3
b = 4

a + b == myName
print(myName)


list_1 = [2, 4, 6, 8, 10]     # even numbers from 2 to 10
list_2 = [1, 3, 5, 7, 9]      # odd numbers from 1 to 9

# 2. Use a for loop to create list_sum
list_sum = []
for i in range(len(list_1)):
    list_sum.append(list_1[i] + list_2[i])

print("list_1:", list_1)
print("list_2:", list_2)
print("list_sum:", list_sum)
