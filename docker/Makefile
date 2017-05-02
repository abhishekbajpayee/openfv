# Makefile to build OpenFV docker images

# TODO: doesn't seem to be working properly
remove-untagged-images:
	@-(docker rmi $$(docker images | grep "^<none>" | awk '{print $3}')) > /dev/null

build-image:
	@docker build -t openfv/openfv .

build-cuda-image: 
	@-(cd cuda && nvidia-docker build -t openfv/openfv-cuda .)

pull-images:
	@docker pull openfv/openfv
	@docker pull openfv/openfv-cuda

push-images:
	@docker push openfv/openfv
	@docker push openfv/openfv-cuda

images: build-image build-cuda-image
