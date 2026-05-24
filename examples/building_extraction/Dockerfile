# 0. from base paddlers image
FROM paddlers:latest

# 1. install mysql and nodejs
RUN apt-get update \
	&& apt-get install -y mysql-server mysql-client libmysqlclient-dev \ 
		git curl \
	&& curl -sL https://deb.nodesource.com/setup_16.x | bash - \
	&& apt-get install -y nodejs

# 2. clone geoview
WORKDIR /opt
RUN git clone --recursive -b release/0.1 https://github.com/PaddleCV-SIG/GeoView.git \
ENV PYTHONPATH /opt/GeoView/PaddleRS

# 3. install backend requirements 
WORKDIR /opt/GeoView/backend
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
	&& mv .flaskenv_template .flaskenv

# 4. install frontend requirements 
WORKDIR /opt/GeoView/frontend
RUN npm install

# 5. set working directory
WORKDIR /opt/GeoView
