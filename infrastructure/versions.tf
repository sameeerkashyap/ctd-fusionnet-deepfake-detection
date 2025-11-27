terraform {
  required_version = ">= 1.5.7, < 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.25.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = ">= 4.0.0"
    }
    local = {
      source  = "hashicorp/local"
      version = ">= 2.5.1"
    }
  }
}

