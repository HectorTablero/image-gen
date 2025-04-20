# TODO



## PAPER PARA SABER LOS PARAMETROS OPTIMOS DE B_min y B_max EN RUIDO LINEAL
Aquí está el enlace al paper original de DDPM (Denoising Diffusion Probabilistic Models) por Ho et al. (2020) donde se establecen estos valores como punto de referencia:
https://arxiv.org/abs/2006.11239
En la sección 4 "Experiments", específicamente en la página 6, los autores describen su configuración:

"We use a linear variance schedule from β₁ = 10⁻⁴ to βₜ = 0.02"

También puedes consultar el paper "Improved Denoising Diffusion Probabilistic Models" por Nichol y Dhariwal (2021), que realizó un estudio más exhaustivo sobre estos parámetros:
https://arxiv.org/abs/2102.09672
En este segundo paper, los autores experimentan con diferentes programaciones de ruido, pero confirman que el rango de valores similar al del paper original funciona bien como punto de partida.
El código de referencia original de los autores también implementa estos valores y está disponible en GitHub.