const swagger = document.getElementById("swagger-ui");

SwaggerUIBundle({
  domNode: swagger,
  url: swagger.dataset.specUrl,
  presets: [SwaggerUIBundle.presets.apis],
  deepLinking: true,
  displayRequestDuration: true,
  docExpansion: "list",
  validatorUrl: null,
  requestInterceptor(request) {
    const contentType = new Headers(request.headers).get("Content-Type") ?? "";
    if (contentType.split(";")[0].trim() === "application/x-ndjson" &&
        typeof request.body === "string" && request.body.trimStart().startsWith('"')) {
      // Swagger UI wraps string examples in JSON quotes for media types containing "json".
      try {
        request.body = JSON.parse(request.body);
      } catch {
        // The API reports invalid input.
      }
    }
    return request;
  },
});
