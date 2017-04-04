# Makefile to build OpenFV docker images

# TODO: doesn't seem to be working properly
remove-untagged-images:
	@-(docker rmi $$(docker images | grep "^<none>" | awk '{print $3}')) > /dev/null

remove-images: remove-untagged-images
	@-(docker rmi -f openfv-base-notebook openfv-minimal-notebook openfv-scipy-notebook) 2> /dev/null

build-image: remove-images
	@-(cd base-no-cuda && docker build -t openfv-base-notebook .)
	@-(cd minimal-notebook && docker build -t openfv-minimal-notebook .)
	@-(cd scipy-notebook && docker build -t openfv-scipy-notebook .)
	@docker build -t openfv/openfv .

build-cuda-image: remove-images
	@-(cd base-notebook && docker build -t openfv-base-notebook .)
	@-(cd minimal-notebook && docker build -t openfv-minimal-notebook .)
	@-(cd scipy-notebook && docker build -t openfv-scipy-notebook .)
	@-(cd cuda && docker build -t openfv/openfv-cuda .)

pull-images:
	@docker pull openfv/openfv
	@docker pull openfv/openfv-cuda

push-images:
	@docker push openfv/openfv
	@docker push openfv/openfv-cuda

images: build-image build-cuda-image remove-images
