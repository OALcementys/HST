#!/bin/sh
env=$1

#Safety Checks
if [ -z "$env" ] || ([ "$env" != "dev" ] && [ "$env" != "preprod" ] && [ "$env" != "prod" ])
then
      echo "Please provide arguments: Usage: ./deploy.sh [ENV_TO_DEPLOY]"
      echo "Supported environments: dev / preprod / prod"
      exit 1
fi

echo "Deployment started for environment: " $env

#Creating necessary variables
# You can change your template configuration file name but you have to keep ${env} in the name
fileName="config.${env}.yml"
# here you have to change the folder name of your lambda env, don't remove the source folder : datatys-dev-lambda-env
appName="lambda-python-example"
sourceLocation="s3://datatys-dev-lambda-env/${appName}/${fileName}"

# Downloading the config file from S3
echo "downloading file from s3"
aws s3 cp $sourceLocation $fileName

#Deploying the lambdas via serverless
echo "Deploying Lambdas"
serverless deploy --stage $env
echo "Lambdas deployed"

#Once done, we should delete the config file from local machine
echo "deleting config file from local"
rm $fileName


echo "Deployment complete for environment: " $env
exit 0