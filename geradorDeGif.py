import os
import glob
import re
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog

def natural_sort_key(filename):
    """
    Cria uma chave de ordenação "natural" para um nome de arquivo.
    Extrai todos os números e usa o último encontrado como a chave principal.
    """
    basename = os.path.basename(filename)
    numbers = re.findall(r'\d+', basename)
    
    if numbers:
        # Usa o último número encontrado como a chave de ordenação
        return int(numbers[-1])
    else:
        return 0

def get_font(size=30):
    """
    Tenta carregar uma fonte TrueType comum.
    Recorre à fonte padrão minúscula se não encontrar.
    """
    try:
        # Tenta carregar Arial (comum no Windows/Mac)
        font = ImageFont.truetype("arial.ttf", size)
    except IOError:
        try:
            # Tenta carregar DejaVuSans (comum no Linux)
            font = ImageFont.truetype("DejaVuSans.ttf", size)
        except IOError:
            # Se tudo falhar, usa a fonte bitmap padrão (muito pequena)
            print("AVISO: Fonte 'arial.ttf' ou 'DejaVuSans.ttf' não encontrada.")
            print("Usando fonte padrão (pode ficar muito pequena).")
            font = ImageFont.load_default()
    return font

def draw_text_with_outline(draw, text, position, font, fill_color="white", shadow_color="black"):
    """
    Desenha texto com um contorno simples (sombra) para melhor visibilidade.
    """
    x, y = position
    # Desenha a "sombra" preta deslocada
    draw.text((x-1, y-1), text, font=font, fill=shadow_color)
    draw.text((x+1, y-1), text, font=font, fill=shadow_color)
    draw.text((x-1, y+1), text, font=font, fill=shadow_color)
    draw.text((x+1, y+1), text, font=font, fill=shadow_color)
    
    # Desenha o texto principal por cima
    draw.text(position, text, font=font, fill=fill_color)

def create_gif(image_files, output_filename, duration_ms=150):
    """
    Pega uma lista de arquivos de imagem, adiciona texto de época e os compila em um GIF.
    """
    if not image_files:
        print(f"Nenhuma imagem encontrada para criar {output_filename}. Pulando.")
        return

    print(f"Processando {len(image_files)} imagens para {output_filename}...")
    
    # Ordena a lista de arquivos
    image_files.sort(key=natural_sort_key)
    
    processed_images = []
    
    try:
        for filename in image_files:
            # Abre a imagem original
            img = Image.open(filename).convert("RGBA")
            
            # Prepara para desenhar na imagem
            draw = ImageDraw.Draw(img)
            
            # --- Adicionar Texto ---
            
            # 1. Pega o número da época do nome do arquivo
            epoch_num = natural_sort_key(filename)
            
            # 2. Decide o texto
            is_ema = "_ema" in os.path.splitext(os.path.basename(filename))[0]
            epoch_text = f"Época: {epoch_num}"
            if is_ema:
                epoch_text += " (EMA)"
            
            # 3. Define fonte e posição
            # Tamanho da fonte dinâmico com base na altura da imagem
            font_size = max(15, int(img.height / 25))
            font = get_font(font_size)
            
            # Posição: Canto inferior esquerdo (com 10px de padding)
            text_x = 10
            text_y = img.height - font_size - 10
            position = (text_x, text_y)

            # 4. Desenha o texto com contorno
            draw_text_with_outline(
                draw, 
                epoch_text, 
                position, 
                font, 
                fill_color="white", 
                shadow_color="black"
            )
            
            # Adiciona a imagem processada à lista
            # Converte para 'P' (modo paletizado) que é o formato GIF
            processed_images.append(img.convert('P', dither=Image.Dither.FLOYDSTEINBERG))

        # --- Salvar o GIF ---
        print(f"Salvando {output_filename}...")
        processed_images[0].save(
            output_filename,
            save_all=True,
            append_images=processed_images[1:],
            duration=duration_ms,
            loop=0,
            optimize=False
        )
        print(f"\n>>> SUCESSO: GIF salvo em {output_filename}\n")
        
    except Exception as e:
        print(f"ERRO ao criar GIF {output_filename}: {e}")
        import traceback
        traceback.print_exc()

# --- Bloco Principal (Idêntico ao anterior) ---
if __name__ == "__main__":
    
    # --- Configuração via Janelas de Diálogo ---
    root = tk.Tk()
    root.withdraw()

    try:
        # 1. Perguntar pela pasta de IMAGENS
        print("Abrindo janela para selecionar a pasta de imagens...")
        image_folder = filedialog.askdirectory(
            title="Passo 1: Selecione a pasta que contém suas imagens"
        )
        if not image_folder:
            print("Seleção de pasta de imagens cancelada. Saindo.")
            exit()
        print(f"Pasta de imagens selecionada: {image_folder}")

        # 2. Perguntar pelo nome e local de SAÍDA
        print("Abrindo janela para definir o nome e local de saída...")
        output_base_path = filedialog.asksaveasfilename(
            title="Passo 2: Escolha o local e o NOME BASE para os GIFs",
            defaultextension=".gif",
            filetypes=[("Arquivo GIF", "*.gif")],
            initialfile="progresso.gif"
        )
        if not output_base_path:
            print("Seleção de arquivo de saída cancelada. Saindo.")
            exit()

    finally:
        root.destroy()

    # --- Processar os nomes de saída ---
    output_dir = os.path.dirname(output_base_path)
    base_filename = os.path.basename(output_base_path)
    base_name = os.path.splitext(base_filename)[0]

    output_gif_normal = os.path.join(output_dir, f"{base_name}_normal.gif")
    output_gif_ema = os.path.join(output_dir, f"{base_name}_ema.gif")

    print("--------------------------------------------------")
    print(f"Salvando GIF normal em: {output_gif_normal}")
    print(f"Salvando GIF EMA em:   {output_gif_ema}")
    print("--------------------------------------------------")

    # Duração de cada frame no GIF (em milissegundos)
    FRAME_DURATION_MS = 500

    # Encontra todos os arquivos de imagem
    search_path_png = os.path.join(image_folder, '*.png')
    search_path_jpg = os.path.join(image_folder, '*.jpg')
    all_files = glob.glob(search_path_png) + glob.glob(search_path_jpg)

    if not all_files:
        print(f"Nenhum arquivo .png ou .jpg encontrado em '{image_folder}'")
        exit()

    # Separa os arquivos em duas listas: normal e ema
    normal_files = []
    ema_files = []

    for f in all_files:
        basename = os.path.basename(f)
        if os.path.splitext(basename)[0].endswith('_ema'):
            ema_files.append(f)
        else:
            if f == output_gif_normal or f == output_gif_ema:
                continue
            normal_files.append(f)

    # --- Gerar os GIFs ---
    create_gif(normal_files, output_gif_normal, duration_ms=FRAME_DURATION_MS)
    create_gif(ema_files, output_gif_ema, duration_ms=FRAME_DURATION_MS)