
/** @type {import('next').NextConfig} */

const nextConfig = {

  images: {

    domains: ['localhost'],

    unoptimized: true,

    remotePatterns: [

      {

        protocol: 'https',

        hostname: '**',

      },

    ],

  },

}
