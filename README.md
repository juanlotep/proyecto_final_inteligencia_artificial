## Informe proyecto final inteligencia artificial
### Resumen – En el presente informe de proyecto, se presenta al lector tanto regresión, en donde se realiza una predicción del valor asociado a casas, junto con clasificación y predicción del número de pisos asociados a estas mismas. Para esto, se realiza métodos de machine learning de tipo supervisado, en donde se emplean soluciones con máquinas de soporte vectorial utilizando PCA, clasificadores ovo y ova, knn, redes neuronales y por último la pseudoinversa de moore-penrose

### Índice de Términos –Regresión, clasificación y predicción, machine learning, maquinas de soporte vectorial, clasificadores (ovo, ova, knn), redes neuronales, pseudoinversa de moore-penrose, PCA.

##### INTRODUCCIÓN
Durante el desarrollo del presente proyecto correspondiente a la predicción del valor asociado a casas, dadas ciertas características, junto con la clasificación y predicción de la cantidad de pisos asociados a estas, fue necesario el uso de un dataset suministrado en “kaggle”, el cual se denomina “Housing-New-Dataset” [1]. Este conjunto de datos contiene características tales como el precio de la casa, área, numero de baños, habitaciones para huéspedes, cantidad de pisos, aire acondicionado y demás atributos. Una vez caracterizado dicho dataset, se procedió a realizar soluciones con métodos distintos supervisados, los cuales fueron explicados a gran detalle en la clase correspondiente a inteligencia artificial. Una vez especificados los distintos aspectos tratados en el presente apartado, se anexa el tratamiento de datos realizado en Colab:

![Texto alternativo](https://user-images.githubusercontent.com/101753582/202934434-40e6b48c-60a1-4c08-9dd8-273f367d1e5f.png)
*Tabla 1. Descripción de los datos.*

![Texto alternativo](https://user-images.githubusercontent.com/101753582/202934564-2e0cb754-da32-4e2f-a1d5-db48742df705.png)
*Ilustración 1. Histogramas de características.*

![Texto alternativo](https://user-images.githubusercontent.com/101753582/202934627-4df0d8ae-5b25-48f1-8191-07ba26ce7c6d.png)
*Ilustración 2. Número de pisos asociados a casas.*

##### OBJETIVOS ESPECIFICOS
- Realizar clasificadores que puedan predecir el número de pisos correspondientes a casas
- Obtener una función por medio de regresión, en donde sea capaz de predecir el valor asociado a casas.
- Analizar y contrarrestar los resultados obtenidos, de tal modo que se pueda obtener la mejor predicción a ciertas variables, de acuerdo o en base a metricas tales como el coeficiente de correlación de matheuws, accuracy, sensibilidad, F1 score y el coeficiente de correlación.

##### SECCIÓN DE DESARROLLO 
Primero que nada, cabe mencionar que debido a que en el dataset utilizado, no presentaba datos de tipo NaNs, no hubo la necesidad que completar ciertas casillas de este mismo. Partiendo de esto, se realizó una caracterización grafica de dichos datos por medio de histogramas en donde se logró observar el comportamiento de estos mismos. Añadido a esto, y teniendo en cuenta que el problema a solucionar es de tipo supervisado, se partió el conjunto de datos para todos los métodos utilizados en grupos de entrenamiento y prueba. Una vez aclarado dichos procesos, se presenta al lector la parte A Y parte B del presente proyecto en donde se realizaron la ejecución de los métodos anteriormente mencionados en el apartado de resumen.

#### *Parte A Clasificación y predicción al número de pisos asociado a casas:*
Para la debida clasificación y predicción a dicho atributo. En primer lugar, se escalizaron los datos, seguido a esto, se realizó PCA en donde pasó de tener 14 componentes (de 0 a 14) a pasar a 10 componentes. Esto gracias a que la varianza explicada fue aproximadamente equivalente al 97%. Esto indica que se perdió un 3% de información, pero a la vez se redujo el numero de componentes lo que facilita la clasificación y complejidad de los métodos. Ahora bien, los métodos que se implementaron fueron: 

- Maquinas de soporte vectorial con kernel RBF. En donde se varió sus hiperparámetros por medio de bucles como (for), o por métodos mas eficientes como lo son *GridSearch*.

- *“OneVsOneClassifier”* y *“OneVsAllClassifier”*, en donde se determinó el óptimo “K” para que dichos métodos cumplieran con las mejores metricas tales como el coeficiente de correlación de matheuws, accuracy, y demás metricas. Teniendo en cuenta las desventajas de cada método pues bien se sabe que en One vs All, la distribución de clases está casi siempre desequilibrada en el conjunto de entrenamiento, y además de esto, la escala de valores de confianza puede diferir. Mientras que por otro lado, para el clasificador One vs One o All vs All, el número de clasificadores aumenta, pero son clasificadores más simples para un K grande. Además de esto, dependiendo de la aplicación, los resultados pueden ser ambiguos.
- Se hizo uso del método Knn, en el cual se varió el hiperparámetro “K” correspondiente al numero de vecinos mas cercanos, con el cual dicho método funciona, Esto teniendo en cuenta las metricas anteriormente mencionadas. Además de ello, también se tuvo en cuenta los problemas que esta solución conlleva, los cuales son:

1. Si se toman muchas muestras X, la clasificación puede llegar a ser lenta.
2. Valores pequeños de K son susceptibles a ruido, mientras que K grandes son más inmunes a ruido, pero si dicho hiperparámetro es exageradamente grande, las categorías con pocas muestras pueden llegar a no ser seleccionadas nunca.

Correspondiente al uso de PCA en el presente problema, se presenta los resultados obtenidos con respecto a esta misma:

![Texto alternativo](https://user-images.githubusercontent.com/101753582/202934955-57ab7bf3-ad66-4118-9ec6-1719833e2d92.png)
*Ilustración 3. Varianza explicada variando el número de componentes para la parte A.*

Como se puede observar, se puede dejar perfectamente las 10 componentes principales, perdiendo un aproximado de un 3% de información de los datos.

#### *Parte B: Regresión y predicción al costo asociado a casas*
Ahora bien, con respecto a la parte de regresión y predicción al costo asociado a casas del presente proyecto, también se escalizaron los datos, esto debido a que por ejemplo características tales como el área, estaba en magnitudes por encima 10^3, y otras variables estaban en binario, luego no se tenía una buena dimensionalidad hablando en base a estas órdenes de magnitud, por lo cual dificulta más el proceso de una buena predicción a dichas variables. Además de esto, cabe mencionar que también se hizo PCA, en donde se dejó las principales componentes, teniendo como referencia la varianza explicada en los datos. Una vez especificados dichos aspectos, se presentan al lector los métodos utilizados tales como:

-	pseudoinversa de moore-penrose. En donde se utiliza matrices como base del presente algoritmo, las cuales contienen a la salida Y, las entradas o las características del dataset junto con una columna de unos “1”. A partir de esto y fundamentándose en algebra lineal, dicha solución esta dada por la siguiente ecuación:  ϕ =((X^T*X)^(-1) ) X^T)Y  [1]
En donde ϕ, son los parámetros a encontrar, X^T es la matriz con los datos iniciales, pero traspuesta, y por último “Y” siendo la salida.

-	Redes neuronales. En este caso se hizo uso del “modelo” de perceptrón multicapa totalmente conectado, en donde se dejo como función de activación la función Relu, y con base en lo visto en la clase de inteligencia artificial, se dejo el modelo con 3 capas ocultas con un total de 50 neuronas en dichas capas, y en la capa de salida con una neurona, debido a que no se está trabajando una clasificación multiclase, y la salida de dicha red neuronal es un número.

-	Correspondiente al uso de PCA en el presente problema, se presenta los resultados obtenidos con respecto a esta misma:


![Texto alternativo](https://user-images.githubusercontent.com/101753582/202935183-2dcdb12d-068b-4f8e-bf0c-3875b3d0fd32.png)
*Ilustración 4. Varianza explicada variando el número de componentes para la parte B.*

Como se puede observar, se puede dejar perfectamente las 10 componentes principales, perdiendo un aproximado de un 3% de información de los datos.

##### RESULTADOS

Teniendo en cuenta lo presentado anteriormente en el apartado de sección de desarrollo, se obtuvo los siguientes resultados, tanto para la parte A como para la parte B:

#### *Parte A Clasificación y predicción al número de pisos asociado a casas:*
-	Máquinas de soporte vectorial con kernel RBF:

Para este método, se realizaron 2 posibles soluciones, variando hiperparámetros por medio de bucles y por medio de GridSearch. De las cuales se muestran a continuación:

A.Por medio de bucles:

![Texto alternativo](https://user-images.githubusercontent.com/101753582/202935312-cece51cb-fd8d-4631-9c34-f7f82cdbf525.png)
*Ilustración 5. ACC variando gamma.*

![Texto alternativo](https://user-images.githubusercontent.com/101753582/202935347-6a5887b8-0e23-42a7-a7c6-5e8e36a6f798.png)
*Ilustración 6. MCC variando gamma.*

![Texto alternativo](https://user-images.githubusercontent.com/101753582/202935375-c8d1e0e6-b0a3-47e3-b6c4-6fbf1c33a757.png)
*Ilustración 7. ACC variando C.*

![Texto alternativo](https://user-images.githubusercontent.com/101753582/202935393-36e61c9e-572f-4ab8-ba28-56a57198f3a6.png)
*Ilustración 8. MCC variando C.*

Se obtuvo la matriz de confusión:

![Texto alternativo](https://user-images.githubusercontent.com/101753582/202935455-1aa5655b-a04c-4c61-85ef-241f55e53ccc.png)
![Texto alternativo](https://user-images.githubusercontent.com/101753582/202935485-7f6ececb-d536-477f-9aac-b0025afee33d.png)
*Tabla 2. Matriz de confusión junto con metricas como coeficiente de Matthews y accuracy, para el método máquinas de soporte*

B. Por medio de GridSearch:

![Texto alternativo](https://user-images.githubusercontent.com/101753582/202935534-d0f0d150-c86b-470a-bd39-825635a1d440.png)
*Tabla 3. Mejores hiperparámetros para máquinas de soporte vectorial, por medio de la búsqueda de GridSearch.*

Se obtuvo la matriz de confusión:

![Texto alternativo](https://user-images.githubusercontent.com/101753582/202935593-cab6b3d1-210a-4604-89fa-c4a9fd9218e7.png)
*Tabla 4. Matriz de confusión para el método máquinas de soporte, por medio de la búsqueda de GridSearch. *

-	OneVsOneClassifier:
-	
Se obtuvo la matriz de confusión junto con metricas tales como el coeficiente de correlación de matthews junto con el accuracy.

![Texto alternativo](https://user-images.githubusercontent.com/101753582/202935684-79bbb0ee-db71-4d4b-b66a-b8bfabe3d5a0.png)
*Tabla 5. Matriz de confusión para el método OneVsOneClassifier, junto con metricas como coeficiente de Matthews y accuracy.*

-	 OneVsAllClassifier:
Se obtuvo la matriz de confusión junto con metricas tales como el coeficiente de correlación de matthews junto con el accuracy.

![Texto alternativo](https://user-images.githubusercontent.com/101753582/202935729-75dcb5d8-2dab-48d9-aa00-8c9eb975ad47.png)
*Tabla 6. Matriz de confusión para el método OneVsAllClassifier, junto con metricas como coeficiente de Matthews y accuracy.*

-	Knn:
Para Knn se varió el hiperparámetro k, tal y como se observa en la siguiente figura, teniendo como métrica el accuracy:
![Texto alternativo](https://user-images.githubusercontent.com/101753582/202935781-e86c5488-3abb-43f8-86d6-3779cd221325.png)
*Ilustración 9. Accuracy variando K.*

Teniendo esto en cuenta, se obtuvo los siguientes resultados:
![Texto alternativo](https://user-images.githubusercontent.com/101753582/202935801-3792b4a8-9be3-49f2-8781-e760452045a5.png)
*Tabla 7. Matriz de confusión para el método Knn, junto con metricas como coeficiente de Matthews y accuracy.*

#### *Parte B: Regresión y predicción al costo asociado a casas:*
-	pseudoinversa de moore-penrose:
Para la pseudoinversa de moore-penrose, se concatenó a la matriz de entrada, una columna de unos “1”, además de esto se estableció como salida el valor de las casas, y se realizó la ecuación [1] mostrada en el apartado de sección de desarrollo, como se muestra a continuación:

![Texto alternativo](https://user-images.githubusercontent.com/101753582/202935866-c7c8d1bf-5be9-4b02-889a-8ff776388ac4.png)

Teniendo esto en cuenta, se obtuvo los siguientes resultados, los cuales se pueden comparar claramente con las salidas del dataset:

![Texto alternativo](https://user-images.githubusercontent.com/101753582/202935896-2cad59db-d1ab-4b09-8c06-d64c17789c71.png)
*Ilustración 10. Salida del dataset.*

![Texto alternativo](https://user-images.githubusercontent.com/101753582/202935920-93d698a8-4b4b-4861-a542-227266eb5176.png)
*Ilustración 11. Resultados de la función de hipotesis.*

Además de esto, se presenta el coeficiente de correlación obtenido con el presente método:

![Texto alternativo](https://user-images.githubusercontent.com/101753582/202935951-5489ce6c-83f8-4bec-9a3a-9924fbbc100d.png)
*Ilustración 12. Coeficiente de correlación.*

-	Redes neuronales, modelo de perceptrón multicapa totalmente conectado:
Para redes neuronales, se realizó el método con y sin usar PCA, con el fin de contrarrestar los resultados, los cuales fueron:
A. Sin PCA:

![Texto alternativo](https://user-images.githubusercontent.com/101753582/202935997-5ae4530c-335e-4309-94b8-9f4e31271914.png)
*Ilustración 13. Conjunto de entrenamiento.*

![Texto alternativo](https://user-images.githubusercontent.com/101753582/202936021-f3601817-f32b-4a78-aabb-cbeb24ae200a.png)
*Ilustración 14. Predicción de hipotesis*

se presenta el coeficiente de correlación obtenido con el presente método:

![Texto alternativo](https://user-images.githubusercontent.com/101753582/202936043-580986e8-2756-4cce-8ea8-c4168e62c72e.png)
*Ilustración 15. Coeficiente de correlación en conjunto de entrenamiento y validación*

B. Con PCA: Cabe mencionar que se utilizó las componentes principales mencionadas al inicio del presente informe. 

![Texto alternativo](https://user-images.githubusercontent.com/101753582/202936074-26bb76e5-3b8e-4ca8-ac3b-ec6036a87ea1.png)
*Ilustración 16. Conjunto de entrenamiento *

Dicho esto, los resultados fueron:
![Texto alternativo](https://user-images.githubusercontent.com/101753582/202936109-eaad1952-fd9d-4a36-80d1-704690e4c07e.png)
*Ilustración 17. Coeficiente de correlación en conjunto de entrenamiento y validación*

##### CONCLUSIONES

- A partir de la varianza explicada, tanto para el problema de regresión como el de clasificación, se concluye que se mejora el resultado, esto a pesar de que de pierda un aproximado de 3% de información en los datos. Pues se logra reducir el problema de 14 componentes principales a 12 componentes principales, lo cual de alguna u otra manera le facilita la clasificación y regresión a los diferentes métodos empleados. 

- A partir de lo realizado en el presente proyecto, se concluye que es fundamental y de suma importancia que los datos estén escalizados, (en este caso con standardscaler),  esto debido a que muchas de las características tenían una varianza bastante grande, es decir, mientras que en  características como el área de la casa estaba en escala de 10^3, se presentaba otras variables que estaba en binario (0 y 1), por ende, al realizar ya sea una regresión por ejemplo, se dificulte realizar el descenso del gradiente u otros métodos. Además de esto, cabe mencionar que al normalizar dichos datos, se le facilita un montón la clasificación a un clasificador, ya que puede llegar más rápido a la solución, es decir, converge en menos tiempo.

- Para la parte correspondiente a la de regresión, se puede concluir que tanto con la solución de pseudoinversa de moore-penrose como con redes neuronales con PCA, se obtiene una predicción de una casa que se podría utilizar en la práctica, mas no es 100% confiable. Esto debido a que a ciencia cierta no obtuvo un coeficiente de correlación muy bajo, pero tampoco fue alto, ya que este fue de 0.69 en el conjunto de entrenamiento, y en prueba fue de 0.58, mientras que para el método utilizado pseudoinversa de moore-penrose fue de 0.683, en todo el conjunto de datos. además de lo anteriormente mencionado, cabe aclarar, que una de las razones del resultado de este coeficiente de tal manera fue debido los datos presentaban ruido, pues bien, se sabe que podrían estar inflados algunos de los costos de ciertas casas. 

- Por último, con respecto a la parte de clasificación y predicción de la cantidad de pisos asociados a una casa, se puede afirmar que hubo una clase desbalanceada la cual, ninguno de los clasificadores implementados pudo predecir dicha clase, es por esto, que no se puede medir un clasificador por medio del accuracy, ya que como se puede observar se obtuvo un accuracy relativamente alto, más sin embargo al revisar en detalle en la matriz de confusión, se encontraba demasiados errores, concretamente en una clase. Teniendo esto en cuenta, se evalua el desempeño de dichos métodos mediante la matriz de confusión y el coeficiente de correlación de Matthews. De tal manera que, se puede decir que el mejor método fue maquinas de soporte vectorial con kernel RBF.

##### BIBLIOGRAFIA

[1]. Kaggle. (12 de 9 de 2020). Obtenido de https://www.kaggle.com/datasets/shrutipandit707/housingnewdataset









































