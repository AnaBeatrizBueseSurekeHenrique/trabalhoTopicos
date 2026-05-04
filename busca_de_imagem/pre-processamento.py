# Load the image
import cv2
import selectivesearch
import matplotlib.pyplot as plt
def pre_processamento(nome_img):
	image_path = nome_img
	image = cv2.imread(image_path)
	# proporsals: cordenadas de onde pode ter área de interesse!
	proposals = selective_search(image)

	output_image = image.copy()
	# essa parte aq é só pra fazer os quadrados pra visualizar onde tão as áreas!
	for (x, y, w, h) in proposals:
		x1, y1 = x, y  
		x2, y2 = x + w, y + h  # Bottom-right corner

		cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

	plt.imshow(output_image)
	plt.show()
	return proposals
def selective_search(image):
	new_height = int(image.shape[1] / 4)
	new_width = int(image.shape[0] / 4)
	resized_image = cv2.resize(image, (new_width, new_height))
	img_lbl, regions = selectivesearch.selective_search(
	resized_image, scale=300, sigma=0.9, min_size=10)
	candidates = set()
	for r in regions:
		# vê se a região n tá repetida
		if r['rect'] in candidates:
			continue  # Skip this region if it's a duplicate
		# vê se aregião n é muto pequena
		if r['size'] < 200:
			continue  

		# Extract the coordinates and dimensions of the region's rectangle
		x, y, w, h = r['rect']

		# se o tamanho ou lagura é = 0, então pula
		if h == 0 or w == 0:
			continue 
		if w / h > 1.2 or h / w > 1.2:
			continue  

		candidates.add(r['rect'])
	candidates_scaled = [(int(x * (image.shape[1] / new_width)),
						int(y * (image.shape[0] / new_height)),
						int(w * (image.shape[1] / new_width)),
						int(h * (image.shape[0] / new_height)))
						for x, y, w, h in candidates]

	return candidates_scaled
