class Tensor_algebra_polynom:
    def __init__(self, string, num_gen):
        self._num_gen = num_gen
        self._string = string.replace("-", "+-")

        self._main_data = self._string.split('+')
        if self._main_data[0] == '':
            self._main_data = self._main_data[1:]
        for i in range(len(self._main_data)):
            self._main_data[i] = self._main_data[i].replace('-', '-*').split('*')
        for i in range(len(self._main_data)):
            if self._main_data[i][0] != '-':
                self._main_data[i] = ['+'] + self._main_data[i]
        for i in range(len(self._main_data)):
            self._main_data[i] = self.canonical_viev(self._main_data[i])
        self._string = string.replace("+-", "-")

    def print_w(self):
        print(self._string)

    def ger_str(self):
        return self._string

    def get_main_data(self):
        return self._main_data

    def print_data(self):
        self._main_data = sorted(self._main_data)
        for i in self._main_data:
            print(i)

    def canonical_viev(self, monom):
        if len(monom) == 2:
            return monom
        if len(monom) == 3:
            if (int(monom[1].split('u_')[1]) - int(monom[2].split('u_')[1])) == 1:
                monom[1], monom[2] = monom[2], monom[1]
                flag = 1
                if monom[0] == '+':
                    monom[0] = '-'
                    flag -= 1
                if monom[0] == '-' and flag == 1:
                    monom[0] = '+'
                return monom
            if int(monom[1][2]) == self._num_gen and int(monom[2][2]) == 1:
                monom[1], monom[2] = monom[2], monom[1]
                flag = 1
                if monom[0] == '+':
                    monom[0] = '-'
                    flag -= 1
                if monom[0] == '-' and flag == 1:
                    monom[0] = '+'
                return monom

            return monom
        elif len(monom) >= 4:
            h_c = 1
            cl_mon = [monom]
            while True:
                for i in cl_mon:
                    temp = []
                    for k in self.transpose(i):
                        # print(self.transpose(i))
                        if k not in cl_mon:
                            temp += [k]
                    cl_mon += temp
                if len(cl_mon) == h_c:
                    break
                h_c = len(cl_mon)
            # print(cl_mon)
            temp = []
            flag = 1
            for i in range(1, self._num_gen + 1):
                t = self._num_gen + 1;
                for j in cl_mon:
                    if str("u_" + str(i)) not in j:
                        flag = 0
                        break
                    if j.index("u_" + str(i)) < t:
                        t = j.index("u_" + str(i))
                        temp = []
                        temp += [j]
                    elif j.index("u_" + str(i)) == t:
                        temp += [j]
                if flag == 1:
                    cl_mon = temp
                flag = 1
            return cl_mon[0]

    def transpose(self, monom):
        a = []
        for i in range(len(monom)):
            if i == 1 and abs((int(monom[1].split('u_')[1]) - int(monom[2].split('u_')[1])) % self._num_gen) == 1:
                b = monom.copy()
                b[1], b[2] = b[2], b[1]
                flag = 1
                if b[0] == '+':
                    b[0] = '-'
                    flag -= 1
                if b[0] == '-' and flag == 1:
                    b[0] = '+'
                if b not in a:
                    a += [b]
            elif i == len(monom) - 1 and abs(
                    (int(monom[len(monom) - 1].split('u_')[1]) - int(
                        monom[len(monom) - 2].split('u_')[1]) % self._num_gen)) == 1:
                b = monom.copy()
                b[len(monom) - 1], b[len(monom) - 2] = b[len(monom) - 2], b[len(monom) - 1]
                flag = 1
                if b[0] == '+':
                    b[0] = '-'
                    flag -= 1
                if b[0] == '-' and flag == 1:
                    b[0] = '+'
                if b not in a:
                    a += [b]
            elif len(monom) - 1 > i > 1:
                if abs((int(monom[i].split('u_')[1]) - int(monom[i + 1].split('u_')[1]) % self._num_gen)) == 1:
                    b = monom.copy()
                    b[i], b[i + 1] = b[i + 1], b[i]
                    flag = 1
                    if b[0] == '+':
                        b[0] = '-'
                        flag -= 1
                    if b[0] == '-' and flag == 1:
                        b[0] = '+'
                    if b not in a:
                        a += [b]
                if abs((int(monom[i].split('u_')[1]) - int(monom[i - 1].split('u_')[1]) % self._num_gen)) == 1:
                    b_ = monom.copy()
                    b_[i], b_[i - 1] = b_[i - 1], b_[i]
                    flag = 1
                    if b_[0] == '+':
                        b_[0] = '-'
                        flag -= 1
                    if b_[0] == '-' and flag == 1:
                        b_[0] = '+'
                    if b_ not in a:
                        a += [b_]
        return a

    def __mul__(self, other):
        b = []
        for i in self._main_data:
            for j in other._main_data:
                a = []
                if i[0] == '+' and j[0] == '-':
                    a += ['-']
                if i[0] == '-' and j[0] == '+':
                    a += ['-']
                if i[0] == '+' and j[0] == '+':
                    a += ['+']
                if i[0] == '-' and j[0] == '-':
                    a += ['+']
                a += i[1:] + j[1:]
                b += [a]
        for i in range(len(b)):
            b[i] = '*'.join(b[i])
        b = ''.join(b)
        b = b.replace("+*", "+")
        b = b.replace("-*", "-")
        if b[0] == '+':
            b = b[1:]
        return Tensor_algebra_polynom(b, self._num_gen)

    def __add__(self, other):
        if other._string[0] == '-':
            return Tensor_algebra_polynom(self._string + other._string, self._num_gen)
        else:
            return Tensor_algebra_polynom(self._string + '+' + other._string, self._num_gen)

    """def reduce(self):
        for i in self._main_data:
            if i[0] == '+':
                if ['-'] + i[1:] in self._main_data:
                    self._main_data.remove(['-'] + i[1:])
                    self._main_data.remove(['+'] + i[1:])
                    continue
            if i[0] == '-':
                if ['+'] + i[1:] in self._main_data:
                    self._main_data.remove(['+'] + i[1:])
                    self._main_data.remove(['-'] + i[1:])
"""

    def reduce(self):
        m = self._main_data.copy()
        for i in m:
            if ['+'] + i[1:] in self._main_data and ['-'] + i[1:] in self._main_data:
                self._main_data.remove(['-'] + i[1:])
                self._main_data.remove(['+'] + i[1:])

    def __sub__(self, other):
        str_1, str_2 = self._string, other._string
        if str_2[0] == '-':
            str_2 = str_2.replace('-', '@')
            str_2 = str_2.replace('+', '-')
            str_2 = str_2.replace('@', '+')
        else:
            str_2 = '+' + str_2
            str_2 = str_2.replace('-', '@')
            str_2 = str_2.replace('+', '-')
            str_2 = str_2.replace('@', '+')
        return Tensor_algebra_polynom(str_1 + str_2, self._num_gen)


def com(a, b, c):
    if c == 1:
        return a * b + b * a
    elif c == 0:
        return a * b - b * a


n = 7
answer = ''

u_1 = Tensor_algebra_polynom("u_1", n)
u_2 = Tensor_algebra_polynom("u_2", n)
u_3 = Tensor_algebra_polynom("u_3", n)
u_4 = Tensor_algebra_polynom("u_4", n)
u_5 = Tensor_algebra_polynom("u_5", n)
u_6 = Tensor_algebra_polynom("u_6", n)
u_7 = Tensor_algebra_polynom("u_7", n)
# u_8 = Tensor_algebra_polynom("u_8", n)
# u_9 = Tensor_algebra_polynom("u_9", n)
#f = com(com(u_1, com(u_4, com(u_6, u_2, 1), 0), 1), com(u_5, u_3, 1), 0) - com(com(com(u_4, u_1, 1), com(u_6, u_2, 1), 0), com(u_5, u_3, 1), 0)
#f = com(com(u_2, com(u_3, com(u_5, u_1, 1), 0), 1), com(u_6, u_4, 1), 0) + com(com(com(u_5, u_2, 1), com(u_3, u_1, 1), 0), com(u_6, u_4, 1), 0) - com(com(u_1, com(u_2, com(u_5, u_3, 1), 0), 1), com(u_6, u_4, 1), 0)
#f = com(com(u_1, com(u_2, com(u_5, u_3, 1), 0), 1), com(u_6, u_4, 1), 0) - com(com(u_2, com(u_3, com(u_5, u_1, 1), 0), 1), com(u_6, u_4, 1), 0) - com(com(com(u_5, u_2, 1), com(u_3, u_1, 1), 0), com(u_6, u_4, 1), 0)
#f.reduce()
#print(f.get_main_data())

"""

d = []


def modify_string(s):
    result = ''
    count = 1
    for char in s:
        if char == '0':
            result += char
        elif char == '1':
            result += str(count)
        count += 1
    return result


def remove_duplicate_zeros(s):
    result = ''
    prev_char = ''
    for char in s:
        if char == '0' and prev_char == '0':
            continue
        result += char
        prev_char = char
    return result


def reverse_string(s):
    return s[::-1]


def split_and_convert(s):
    substrings = s.split('0')
    result = [list(substring) for substring in substrings]

    if s[-1] != '0' and len(result) > 1:
        result[0] += result.pop()

    return result


def remove_empty_lists(lst):
    return [sublist for sublist in lst if sublist]


def count_ones(s):
    return s.count('1')


def convert_to_int(arrays):
    return [[int(num) for num in array] for array in arrays]


def process_arrays(arrays):
    max_val = -100
    max_array = None
    min_val = []
    merged_array = []
    answer = []
    if len(arrays) <= 1:
        return []

    for array in arrays:
        array_max = max(array)
        if array_max > max_val:
            max_val = array_max
            max_array = array

    for array in arrays:
        if array != max_array:
            min_val += [min(array)]

    for i in range(len(min_val)):
        merged_array = []
        for array in arrays:
            merged_array += array
        merged_array.remove(max_val)
        merged_array.remove(min_val[i])
        answer += [sorted(merged_array) + [max_val] + [min_val[i]]]
    return answer


def flatten_2d_list(lst):
    return [item for sublist in lst for item in sublist]


def c_(a):
    s = ''
    a = a[::-1]
    for i in range(0, len(a) - 1):
        if i == 0:
            g = com(Tensor_algebra_polynom("u_" + str(a[1]), n), Tensor_algebra_polynom("u_" + str(a[0]), n), (i + 1) % 2)
            s = "[u_" + str(a[1]) + ',' + "u_" + str(a[0]) + "]"
        if i >= 1:
            g = com(Tensor_algebra_polynom("u_" + str(a[i + 1]), n), g, (i + 1) % 2)
            s = "[u_" + str(a[i + 1]) + ',' + s + "]"
    #print(s)
    return [g, s]


for i in range(2 ** n):
    l = reverse_string('0' * (n - len(bin(i)[2:])) + bin(i)[2:])
    if 1 < count_ones(l) < n - 1:
        d += [l]

for i in range(len(d)):
    d[i] = modify_string(d[i])
    d[i] = remove_duplicate_zeros(d[i])
    d[i] = split_and_convert(d[i])
    d[i] = remove_empty_lists(d[i])
    d[i] = convert_to_int(d[i])
    d[i] = process_arrays(d[i])

g = []
for i in range(len(d)):
    if len(d[i]) != 0:
      g += [d[i]]

d = flatten_2d_list(g)
print(len(d))

for i in range(len(g)):
    g[i] = c_(g[i])
    g[i][0].reduce()
    if not g[i][0].get_main_data():
        g[i] = 0

d = []
for i in range(len(g)):
    if g[i] != 0:
        d += [g[i]]

print(len(d))
print(d)
"""

def O(A, B):
    res = 0
    for i in A:
        for j in B:
            if i > j:
                res += 1
    return res


def c(a, u_i):
    g = u_i
    s = u_i.ger_str()
    a = sorted(a)
    a = list(reversed(a))
    for i in range(0, len(a)):
        g = com(Tensor_algebra_polynom("u_" + str(a[i]), n), g, (i + 1) % 2)
        s = "[u_" + str(a[i]) + ',' + s + "]"
    print(s)
    return [g, s]

Set_of_sum = []

for i in range(1, n - 2):
    if i == 2:
        print()
    M = list(range(1, n + 1))
    M.remove(i)
    M.remove(i + 1)
    for j in range(2 ** (n - 2)):
        l = []
        temp = j
        for k in range(n - 2):
            l += [temp % 2]
            temp //= 2
        A, B = [], []
        for k in range(n - 2):
            if l[k] == 1:
                A += [M[k]]
            elif l[k] == 0:
                B += [M[k]]

        d = 0
        if A != [] and B != []:
            if max(A) > i and max(B) > i + 1:
                temp = (len(A) + 1) * (len(B) + 1)
                print(A)
                print(B)
                f, g = c(A, Tensor_algebra_polynom("u_" + str(i), n)), c(B, Tensor_algebra_polynom("u_" + str(i + 1), n))
                if temp % 2 == 0:
                    d = f[0] * g[0] - g[0] * f[0]
                    d.reduce()
                if temp % 2 == 1:
                    d = f[0] * g[0] + g[0] * f[0]
                    d.reduce()
                if d.get_main_data():
                    Set_of_sum += [[d, (-1) ** (O(A, B) + len(A)), '[' + f[1] + ',' + g[1] + ']']]
                    answer += '[' + f[1] + ',' + g[1] + ']'
                    print(1)
    print('\n')

answer = ''
if Set_of_sum[0][1] == -1:
    for i in range(len(Set_of_sum)):
        Set_of_sum[i][1] *= -1
h = Set_of_sum[0][0]
for i in range(len(Set_of_sum) - 1):
    if Set_of_sum[i + 1][1] == 1:
        h += Set_of_sum[i + 1][0]
    elif Set_of_sum[i + 1][1] == -1:
        h -= Set_of_sum[i + 1][0]

h.reduce()
print(h.get_main_data())

for i in range(len(Set_of_sum)):
    a = ''
    if Set_of_sum[i][1] == 1:
        a = '+'
    if Set_of_sum[i][1] == -1:
        a = '-'
    answer += a + Set_of_sum[i][2]
    print(a + Set_of_sum[i][2], i + 1)
print(answer)
print(len(Set_of_sum))

"""
def seacrch_centr(input_string):
    position = 1
    balance = 0
    for char in input_string[1:-1]:
        if char == '[':
            balance += 1
        elif char == ']':
            balance -= 1

        if balance == 0 and char == ',':
            return position
        position += 1


def apply_jacobi_identity(input_string):
    posit = seacrch_centr(input_string)

    x = input_string[1:posit]
    yz = input_string[posit + 1:-1]

    posit = seacrch_centr(yz)

    y = yz[1:posit]
    z = yz[posit + 1:-1]
    deg_x, deg_y = x.count('u'), y.count('u')
    if (-1) ** (deg_x * deg_y) == 1:
        return '[[' + x + ',' + y + '],' + z + ']+[' + y + ',[' + x + ',' + z + ']]'
    if (-1) ** (deg_x * deg_y) == -1:
        return '[[' + x + ',' + y + '],' + z + ']-[' + y + ',[' + x + ',' + z + ']]'


# Пример использования

input_string = '[u_4,[u_6,u_2]]'
output = apply_jacobi_identity(input_string)
print(output)
"""
