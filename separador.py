# a = open('ratings.csv','r')
# x = a.readline(500000)

contador = 0
arq = open('novo.csv','a')
with open("ratings.csv") as infile:
    print(infile)
    for line in infile:
        l = line.split(',')
        user1 = int(l[0])
        user2 = int(l[1])
        if(user1 < 15001 and user2 < 15001):
            arq.write(line)
            contador+=1
        if contador == 50000000:
            break
arq.close()
