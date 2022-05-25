
# copy to s3 and update lambda function

##make sure to install serverless and docker before deploying the service
"""
to install serverless run the following command
# npm install -g serverless

## script deploy
bash deploy.sh  <project-name> \\
<image-name> eneos-hst \\
<aws_id> 448319106780   \\
<port> 8080:8080 \\
<aws-region> eu-west-3\\
<ecr-repo> eneos-ecr\\
<env> dev\\

"""
env=$1
fileName="config.${env}.yml"

# start off by creating a serverless project
sls create — template aws-python3 — path {project-name}
## make sure that your lambda is flawlessly deployed locally
sls invoke local  --path {json path} --function {function name}
echo "status testing lambda locally "
## building docker image based on Dockerfile in current directory
## build your docker onto a containner to make sure it works... if it runs on docker it'll run on any other remote plateform
docker build -t {image-name} .
echo "docker image built  "
## run your docker
docker run -p <port>  {image-name} .

##tag your image as a aws ecr format to be recognized
docker tag {image-name} :latest {aws_id}.dkr.ecr.{aws-region}.amazonaws.com/{image-name}:latest
##create ecr repo
aws ecr create-repository --repository-name {ecr-repo}
## login into ecr
aws ecr get-login-password --region {aws-region} | docker login --username AWS --password-stdin {aws_id}.dkr.ecr.eu-west-3.amazonaws.com
echo "login aws ecr successfully "
##push onto ur repo
docker push {aws_id}.dkr.ecr.eu-west-3.amazonaws.com/{image-name}:latest

##deploy yout lambda
sls deploy
echo "lambda deployed successfully "
##test lambda
sls invoke --path {json path} --function {function name}
