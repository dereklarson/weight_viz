# Start with the lightweight Alpine distro
FROM node:19.2.0-alpine3.15
RUN npm install -g npm@9.2.0

WORKDIR app/
ADD package.json package.json
ADD package-lock.json package-lock.json
ADD dist dist
RUN npm install

EXPOSE 3000
CMD [ "npx", "serve", "dist" ]
