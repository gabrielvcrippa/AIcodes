import copy

class Estado:
    def __init__(self, garcom_pos, garcom_carga, bebidas, limpeza):
        self.garcom_pos = garcom_pos
        self.garcom_carga = garcom_carga
        self.bebidas = bebidas
        self.limpeza = limpeza

    def __str__(self):
        return f"pos={self.garcom_pos} | carga={self.garcom_carga} | bebidas={self.bebidas} | limpeza={self.limpeza}"

    def mover(self, destino):
        if self.garcom_pos != destino:
            return Estado(destino, self.garcom_carga, self.bebidas, self.limpeza)

    def pegar_bebida(self):
        if self.garcom_pos == 'bar' and self.garcom_carga is None:
            return Estado(self.garcom_pos, 1, self.bebidas, self.limpeza)

    def servir_bebida(self):
        if self.garcom_carga is not None and self.garcom_pos != 'bar':
            mesa = int(self.garcom_pos[4])
            if self.bebidas[mesa - 1] > 0:
                novas_bebidas = copy.copy(self.bebidas)
                novas_bebidas[mesa - 1] -= 1
                return Estado(self.garcom_pos, None, novas_bebidas, self.limpeza)

    def limpar(self):
        if self.garcom_carga is None and self.garcom_pos != 'bar':
            mesa = int(self.garcom_pos[4])
            if self.limpeza[mesa - 1] != 0:
                nova_limpeza = copy.copy(self.limpeza)
                nova_limpeza[mesa - 1] = 0
                return Estado(self.garcom_pos, self.garcom_carga, self.bebidas, nova_limpeza)



# Testando o modelo livremente  
problema1 = Estado('bar', None, [0, 2, 0, 0], [0, 0, 1, 1])
print(problema1)
teste1 = problema1.mover('mesa1').mover('bar').pegar_bebida().mover('mesa2').servir_bebida()
print(teste1)
