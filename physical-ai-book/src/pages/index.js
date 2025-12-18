import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            to="/docs/intro">
            Start Learning
          </Link>
        </div>
      </div>
    </header>
  );
}

function Feature({title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="card">
        <div className="card__body text--center padding-horiz--md">
          <h3>{title}</h3>
          <p>{description}</p>
        </div>
      </div>
    </div>
  );
}

const FeatureList = [
  {
    title: 'ROS 2 Fundamentals',
    description: 'Master ROS 2 (Humble) as the nervous system for physical AI robots. Learn nodes, topics, services, and actions.',
  },
  {
    title: 'Simulation to Reality',
    description: 'Build and test robots in Gazebo simulation before deploying to real hardware. Includes URDF, physics, and sensors.',
  },
  {
    title: 'AI Integration',
    description: 'Connect Large Language Models and Vision-Language-Action models to robot control for intelligent autonomy.',
  },
];

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Learn Physical AI and Humanoid Robotics with ROS 2">
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              {FeatureList.map((props, idx) => (
                <Feature key={idx} {...props} />
              ))}
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
