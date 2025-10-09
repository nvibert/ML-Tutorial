# Use nginx alpine for lightweight web server
FROM nginx:alpine

# Copy webapp files to nginx html directory
COPY index.html /usr/share/nginx/html/
COPY assets/ /usr/share/nginx/html/assets/

# Create custom nginx configuration to handle API proxy with CORS
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]