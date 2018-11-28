Alunos: Igor Sodre
        Pedro Pio

SO: Linux Ubuntu 16.04
kernel: 4.8.0-58-generic
distribuição do python: Python 3.6.5 :: Anaconda, Inc.
versao do opencv: 3.1.0
versao do numpy: 1.14.3

Este programa tem como objetivo implementar a transformação de uma imagem de  digital escaneada em um dispositivo touchless
para uma imagem semelhante ao de um dispositivo touch.

Este foi implementado com python 3 juntamente com o auxilio do conjunto de softwares NBIS

antes de executar o programa certifique-se de ter instalado em seu computador o software NBIS juntamente com as seguintes bibliotecas
de python:

bibliotecas         instalação
nunpy          - pip install numpy
scikit-learn   - pip install scikit-learn
scikit-image   - pip install scikit-image
opencv         - pip install opencv-python

para a instalação do software NBIS no sistema ubuntu siga os seguintes passos que foram criados pela colega de disciplina Ana Paula(aptarchetti@gmail.com):

Baixar NBIS_5.0.0: http://nigos.nist.gov:8080/nist/nbis/nbis_v5_0_0.zip
Unzip nbis_v5_0_0.zip
No terminal:
$ sudo apt-get install cmake libc6-dev libc6-dev-i386 g++-multilib
$ sudo apt-get install libx11-dev
$ sudo mkdir /usr/local/NBIS/Main
$ cd Rel_5.0.0
$ ./setup.sh /usr/local/NBIS/Main --64 #(or --32)
$ sudo make config
$ sudo make it
$ sudo make install LIBNBIS=yes
$ export PATH =$PATH:/usr/local/NBIS/Main/bin

com todas as dependências instaladas rode o programa com o comando

python3 trabalho.py

como saida o programa mostra a nota das classificações.

