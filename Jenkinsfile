pipeline {
    agent any

    stages {
        stage('Run train & test env') {
            steps {
                sh 'docker run -dit --name train_test_model train-test-env'
            }
        }
        
        stage('Doing data preprocessing') {
            steps {
                sh 'docker container exec train_test_model python3 preprocessing.py'
            }
        }
        
        stage('Training model') {
            steps {
                sh 'docker container exec train_test_model python3 train.py'
            }
        }
        
        stage('Testing model') {
            steps {
                sh 'docker container exec train_test_model python3 test.py'
            }
        }
        
        stage('Validating results and saving model files') {
            steps {
                script {
                    // Leer el valor de 'accuracy' del archivo JSON utilizando jq
                    def accuracy = sh(script: "docker container exec train_test_model jq '.test_acc' ./results/test_metadata.json", returnStdout: true).trim()
                    
                    // Convertir el valor obtenido a un número
                    accuracy = accuracy.toFloat()

                    // Definir el valor límite
                    def accuracyLimit = 0.8

                    // Mostrar el valor obtenido y el límite
                    echo "Accuracy obtenido: ${accuracy}"
                    echo "Límite de accuracy: ${accuracyLimit}"

                    // Comparar el valor de accuracy con el límite
                    if (accuracy >= accuracyLimit) {
                        echo "Accuracy es mayor o igual al límite. Continuando el pipeline..."
                    } else {
                        error "Accuracy es menor que el límite. Falla el pipeline."
                    }
                }
            }
        }
    }
    
    // Ejecutar una acción solo si el pipeline no ha fallado
    post {
        success {
            // Aquí puedes ejecutar un comando de Docker solo si el pipeline fue exitoso
            echo "Pipeline exitoso, ejecutando comando de Docker..."
            sh 'docker cp train_test_model:./model/class_model.joblib ./class_model.joblib'
            sh 'docker cp train_test_model:./model/tfidf.joblib ./tfidf.joblib'
            sh 'docker rm -f train_test_model'
            sh 'docker restart mlops-poc-fastapi-1'
        }
        failure {
            echo "El pipeline falló, no se ejecutará el comando de Docker."
            sh 'docker rm -f train_test_model'
        }
    }
}
