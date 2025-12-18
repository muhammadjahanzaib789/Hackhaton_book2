// @ts-check
// Docusaurus configuration for Physical AI & Humanoid Robotics Book

import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'Embodied Intelligence in the Real World',
  favicon: 'img/favicon.ico',

  // GitHub Pages deployment configuration
  url: 'https://muhammadjahanzaib789.github.io',
  baseUrl: '/physical-ai-book/',
  organizationName: 'muhammadjahanzaib789',
  projectName: 'physical-ai-book',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          editUrl: 'https://github.com/your-org/physical-ai-book/tree/main/',
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
        },
        blog: false, // Disable blog
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Social card image
      image: 'img/social-card.png',

      navbar: {
        title: 'Physical AI Book',
        logo: {
          alt: 'Physical AI Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'bookSidebar',
            position: 'left',
            label: 'Book',
          },
          {
            href: 'https://github.com/muhammadjahanzaib789/Hackhaton_book2',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },

      footer: {
        style: 'dark',
        links: [
          {
            title: 'Learn',
            items: [
              {
                label: 'Introduction',
                to: '/docs/intro',
              },
              {
                label: 'ROS 2 Fundamentals',
                to: '/docs/chapter-01-ros2-fundamentals',
              },
              {
                label: 'Capstone Project',
                to: '/docs/chapter-08-capstone',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'GitHub Discussions',
                href: 'https://github.com/muhammadjahanzaib789/Hackhaton_book2',
              },
              {
                label: 'ROS Discourse',
                href: 'https://discourse.ros.org/',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/muhammadjahanzaib789/Hackhaton_book2',
              },
              {
                label: 'ROS 2 Documentation',
                href: 'https://docs.ros.org/en/humble/',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI Book Contributors. Built with Docusaurus.`,
      },

      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['python', 'bash', 'yaml', 'json', 'markup'],
      },

      // Table of contents configuration
      tableOfContents: {
        minHeadingLevel: 2,
        maxHeadingLevel: 4,
      },

      // Announcement bar (optional)
      announcementBar: {
        id: 'wip_notice',
        content: 'This book is under active development. Contributions welcome!',
        backgroundColor: '#fafbfc',
        textColor: '#091E42',
        isCloseable: true,
      },
    }),
};

export default config;
