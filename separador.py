arq = open('novo.csv','w')
contador = 1
with open("BX-Book-Ratings.csv") as infile:
    for line in infile:
        arq.write(line)
        print(contador)
        if contador == 15000:
            break
        contador += 1
arq.close()

# maxi = 0
# ele = ''
# with open("novo.csv") as infile:
#     for line in infile:
#         linha = line.split(';')
#         e = len(linha[1])
#         print(linha[1],e)
#         if e > maxi:
#             ele = linha[1]
#             maxi = e
# print(maxi)
# print(ele)