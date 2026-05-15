# Dominant Colours

A web-based dominant colour extraction application.

This application allows users to upload an image and extract a small set of representative colours from it. It returns each dominant colour as a HEX value, together with its proportion in the image. Users can also generate partition masks to visualise which areas of the image correspond to each extracted colour.

*Live app*: https://colours-delta.vercel.app/

---

## Overview

Dominant Colours is a lightweight web interface for analysing the colour composition of images. It is designed for visual research, design analysis, and exploratory image-based projects where understanding the distribution of colours is useful.

The application supports two extraction modes:

- **Auto (Elbow)**: automatically estimates the number of colour clusters.
- **Manual**: allows the user to specify the number of clusters, from 2 to 10.

After processing an image, the app displays:

- dominant colour HEX codes;
- the ratio of each colour in the image;
- optional partition masks for each colour cluster.

## Features

- Upload an image from the browser
- Extract dominant colours from the uploaded image
- Display HEX colour values
- Display the percentage ratio of each colour
- Choose between automatic and manual cluster selection
- Generate partition masks on demand
- Technical explanation page describing the extraction pipeline
- Simple web interface with no user account or gallery system

## How it works

The application extracts representative colours from an image using a clustering-based pipeline.

In brief, the process is:

1. The uploaded image is sent to the extraction API.
2. The image is resized to reduce computation.
3. Pixel values are converted into HSV colour space.
4. Pixels are clustered using K-Means.
5. Each cluster centre is treated as a dominant colour.
6. The ratio of each cluster is calculated.
7. Optional partition masks can be generated for visual inspection.

In **Auto (Elbow)** mode, the app evaluates different values of `k` and selects an appropriate number of clusters using an elbow/knee detection method. In **Manual** mode, the user specifies the value of `k` directly.

For a more detailed explanation, see the “How it works” page in the application.

## Tech stack

- [Next.js](https://nextjs.org/)
- [React](https://react.dev/)
- [TypeScript](https://www.typescriptlang.org/)
- [Tailwind CSS](https://tailwindcss.com/)

The front end communicates with a separate colour extraction API via the `NEXT_PUBLIC_API_BASE_URL` environment variable.

## Getting started

Follow this instruction if you want to 

### Prerequisites

Make sure you have Node.js and npm installed.

### Installation

Clone this repository:

```bash
git clone https://github.com/nanothegigante/colours_web.git
cd colours_web
```

Install dependencies:

```bash
npm install
```

Create a `.env.local` file in the project root and set the API base URL:

```bash
NEXT_PUBLIC_API_BASE_URL=https://colours-api.onrender.com
```
⚠︎ See more details of this API → [colours_api](https://github.com/nanothegigante/colours_api.git)

Start the development server:

```bash
npm run dev
```

Open the app in your browser:

```bash
http://localhost:3000
```

### Available scripts:

```bash
npm run dev
```
Runs the development server.

```bash
npm run build
```
Builds the application for production.

```bash
npm run start
```
Starts the production server after building.

```bash
npm run lint
```
Runs ESLint.


## Project structure

```
colours_web/
├── app/
│   ├── how-it-works/
│   │   └── page.tsx
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx
├── public/
├── package.json
├── next.config.ts
├── tsconfig.json
└── README.md
```

### Environment variables

Example:
```
NEXT_PUBLIC_API_BASE_URL=https://colours-api.onrender.com
```

## Other

### Privacy note
This application does not store uploaded images.

Images uploaded by users are sent to the extraction API only for the purpose of processing and colour analysis. The application does not save uploaded image files, retain them in a gallery, associate them with user accounts, or use them for any secondary purpose.

The app also does not provide user accounts or persistent image management. Once the colour extraction process is complete, the uploaded image data is not stored by this application.

### Author
Built by [nano the gigante](www.nanothegigante.com)↗︎.

