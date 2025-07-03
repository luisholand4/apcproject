def clear_string(string: str) -> str:
    string = string.strip()
    if string.startswith('"'):
        string = string[1:]
    else:
        string = string
    if string.endswith('"'):
        string = string[:-1]
    else:
        string = string
    string = string.replace('""', '"')
    return string

def ler_csv(csv_arq): 
    tabela = [] 
    with open(csv_arq, 'r', encoding='utf-8') as fp:
    for linha in fp: 
        linha = linha.strip()
        if not linha:
            continue
        valorinicial, titulo = linha.split(',',1)
        titulo = clear_string(titulo)
        tabela.append((int(valorinicial), titulo))
    return tabela

def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                custo = 0
            else:
                custo = 1
                
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + custo
            )
            
    return dp[m][n]

def knn(entrada_treino, dados_treino, k): 
    distancias = []
    
    for valorinicial, titulo_treino in dados_treino:
        d = levenshtein(entrada_treino, titulo_treino)
        distancias.append((d, valorinicial))
        
    distancias.sort()
    vizinhos = [valorinicial for _, valorinicial in distancias[:k]]
    
    cont_0 = vizinhos.count(0)
    cont_1 = vizinhos.count(1)
    
    if cont_1 > cont_0:
        return 1
    else:
        return 0


caminho_treino, caminho_teste, k = input().split()
k = int(k)
dados_treino = ler_csv(caminho_treino)
dados_teste = ler_csv(caminho_teste)

valoresiniciais_reais = []
valoresiniciais_preditos = []

for valorinicial, titulo in dados_teste:
    predicao = knn(titulo, dados_treino, k)
    valoresiniciais_reais.append(valorinicial)
    valoresiniciais_preditos.append(predicao)

TP = 0
TN = 0
FP = 0
FN = 0

for real, pred in zip(valoresiniciais_reais, valoresiniciais_preditos):
    if real == 1 and pred == 1:
        TP += 1
    elif real == 0 and pred == 0:
        TN += 1
    elif real == 0 and pred == 1:
        FP += 1
    elif real == 1 and pred == 0:
        FN += 1

total = TP + TN + FP + FN
accuracy = (TP + TN) / total if total else 0
precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

print("+-------------------+")
print("|  Metric   | Value |")
print("+-------------------+")
print(f"| Accuracy  | {accuracy:.2f}  |")
print(f"| Precision | {precision:.2f}  |")
print(f"| Recall    | {recall:.2f}  |")
print(f"| F1-Score  | {f1_score:.2f}  |")
print("+-------------------+")
