FROM golang:1.24 AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /search .

FROM gcr.io/distroless/static
COPY --from=build /search /search
EXPOSE 8000
ENTRYPOINT ["/search"]
